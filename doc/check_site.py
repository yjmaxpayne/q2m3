"""Check bilingual page paths, language switches, code, formulas, and local links."""

from __future__ import annotations

import argparse
import json
import posixpath
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    """Read only the HTML fields needed for delivery checks."""

    def __init__(self, path: Path):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.links = []
        self.switches = {}
        self.protected = []
        self.captures = []
        self.references = []
        self.labels = []
        self.chrome = []
        self.ignore_text = False
        self.viewcode_depth = 0
        self.article = False
        self.language = None
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "html":
            self.language = attrs.get("lang")
        if tag == "article":
            self.article = True
        if tag in ("script", "style"):
            self.ignore_text = True
        if self.viewcode_depth:
            if tag not in ("img", "br", "hr", "input", "link", "meta", "wbr"):
                self.viewcode_depth += 1
        elif {"viewcode-link", "viewcode-back"}.intersection(classes):
            self.viewcode_depth = 1
        for name in ("aria-label", "placeholder", "alt", "title"):
            if attrs.get(name):
                self.labels.append(attrs[name])
        for attr in ("href", "src"):
            if attrs.get(attr):
                self.links.append(attrs[attr])
        if tag == "a" and "hreflang" in attrs:
            self.switches[attrs["hreflang"]] = attrs.get("href", "")
        if self.article and tag == "a" and "internal" in classes:
            self.references.append(attrs.get("href", ""))
        if tag == "pre" or (tag == "dt" and "sig" in classes) or "math" in classes:
            self.captures.append((tag, []))

    def handle_endtag(self, tag):
        if self.captures and self.captures[-1][0] == tag:
            kind, chunks = self.captures.pop()
            self.protected.append((kind, "".join(chunks).replace("¶", "").strip()))
        if tag == "article":
            self.article = False
        if tag in ("script", "style"):
            self.ignore_text = False
        if self.viewcode_depth:
            self.viewcode_depth -= 1

    def handle_data(self, data):
        if not self.viewcode_depth:
            for _, chunks in self.captures:
                chunks.append(data)
        if not self.article and not self.ignore_text:
            self.chrome.append(data)


def resolve_link(page: str, url: str) -> tuple[str, str] | None:
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc:
        return None
    path = unquote(parsed.path)
    if path.startswith("/q2m3/"):
        target = posixpath.normpath(path.removeprefix("/q2m3/"))
    elif path.startswith("/"):
        target = posixpath.normpath(path).lstrip("/")
    else:
        target = posixpath.normpath(posixpath.join(posixpath.dirname(page), path)) if path else page
    return target, unquote(parsed.fragment)


def check_site(root: Path, baseline: Path) -> list[str]:
    """Return delivery errors. External network links require a separate linkcheck."""
    pages = {p.relative_to(root).as_posix(): Page(p) for p in root.rglob("*.html")}
    english = {p for p in pages if not p.startswith("zh_CN/")}
    chinese = {p.removeprefix("zh_CN/") for p in pages if p.startswith("zh_CN/")}
    expected = set(json.loads(baseline.read_text())["pages"])
    ui_messages = json.loads((Path(__file__).parent / "ui-messages.json").read_text())
    errors = []
    if english != expected:
        errors.append(
            f"English page paths changed: missing={sorted(expected - english)}, "
            f"extra={sorted(english - expected)}"
        )
    if english != chinese:
        errors.append(
            f"Language page paths differ: missing={sorted(english - chinese)}, "
            f"extra={sorted(chinese - english)}"
        )
    for relative, page in pages.items():
        is_chinese = relative.startswith("zh_CN/")
        plain = relative.removeprefix("zh_CN/")
        if page.language != ("zh-CN" if is_chinese else "en"):
            errors.append(f"{relative}: wrong HTML language: {page.language}")
        for language, prefix in (("en", ""), ("zh-CN", "zh_CN/")):
            href = page.switches.get(language, "")
            parsed = urlsplit(href)
            resolved = resolve_link(relative, href)
            if not href or parsed.query or parsed.fragment or resolved != (prefix + plain, ""):
                errors.append(f"{relative}: invalid {language} switch: {href!r}")
        for link in page.links:
            resolved = resolve_link(relative, link)
            if resolved is None:
                continue
            target, anchor = resolved
            target_path = root / target
            if not target_path.is_file():
                errors.append(f"{relative}: missing local target: {link}")
            elif anchor and target in pages and anchor not in pages[target].ids:
                errors.append(f"{relative}: missing anchor: {link}")
        if is_chinese and plain in pages:
            chrome = " ".join("".join(page.chrome).split())
            for message in ui_messages:
                text = re.sub(r"<[^>]+>", "", unescape(message))
                text = re.split(r"%\([^)]+\)[sd]", text)[0].strip()
                if text in chrome or any(
                    label == text or label.startswith(text + " ") for label in page.labels
                ):
                    errors.append(f"{relative}: untranslated interface message: {message!r}")
            if page.protected != pages[plain].protected:
                errors.append(f"{relative}: code, signature, or formula differs from English")
            if page.references != pages[plain].references:
                errors.append(f"{relative}: internal reference targets differ from English")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "build/html")
    parser.add_argument(
        "--baseline", type=Path, default=Path(__file__).parent / "site-baseline.json"
    )
    args = parser.parse_args()
    errors = check_site(args.root, args.baseline)
    for error in errors:
        print(error)
    print(f"Site check {'failed' if errors else 'passed'}: {len(errors)} error(s).")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
