# CI runner policy

q2m3 CI uses a runner selector instead of hard-coding GitHub-hosted runners.

## Selection rule

`select-runner.yml` queries the repository's registered runners and selects:

1. `[self-hosted, Linux, X64]` when a runner with all three labels is online;
2. `ubuntu-latest` when no matching runner is online;
3. `ubuntu-latest` when the runner lookup fails;
4. `ubuntu-latest` for pull requests from forks.

The selector itself always runs on `ubuntu-latest`. Jobs consume its JSON output:

```yaml
jobs:
  select_runner:
    uses: ./.github/workflows/select-runner.yml
    secrets:
      SELF_HOSTED_RUNNER_READ_TOKEN: ${{ secrets.SELF_HOSTED_RUNNER_READ_TOKEN }}

  test:
    needs: select_runner
    runs-on: ${{ fromJSON(needs.select_runner.outputs.runner) }}
```

## Required secret

Listing repository runners requires Administration read permission. `GITHUB_TOKEN`
does not grant it, so configure a repository secret named
`SELF_HOSTED_RUNNER_READ_TOKEN` with a fine-grained token that has:

- Repository access: only `yjmaxpayne/q2m3`;
- Repository permissions: Administration → Read-only.

Without this secret the selector logs a warning and every job uses
`ubuntu-latest`. The selector never fails the workflow because of a lookup error.

## Scope

Self-hosted preference applies to test, documentation, and release-build jobs.
Publishing to PyPI, attaching release assets, deploying GitHub Pages, and release
drafting stay on GitHub-hosted runners.
