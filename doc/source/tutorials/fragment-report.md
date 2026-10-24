# Read A Fragment Interaction Report

This page is for experimental partners. You do not run q2m3 or write code. The
computational side runs the workflow and sends you a Markdown report. The
report is in Chinese. This page explains each section of one real report.

## The Example Report

The example report comes from the water-dimer run in
[Fragment Interaction Energy](fragment-interaction.md). That run uses the mode
`full` and the cc-pVDZ basis. Download the report:
{download}`water-dimer-report.md`.

```{literalinclude} water-dimer-report.md
:language: markdown
```

## Title And Fixed Statement

Every report starts with the title `片段相互作用能报告` (fragment interaction
energy report) and the fixed statement `q2m3 不对 Ks 做统计结论。`. The
statement means that q2m3 draws no statistical conclusion about Ks. q2m3 has
no code that computes a correlation coefficient or a p-value.

One report covers all fragment results of one call. Each fragment has its own
section. The section heading gives the fragment name, the mode, and the basis,
for example `片段 water-dimer（模式 full，基组 cc-pvdz）`.

## Interaction Energy

The table `相互作用能 ΔE（counterpoise 校正）` has one row for each method
layer. ΔE is E(AB) − E(A) − E(B). AB is the dimer, and A and B are the two
fragments. The three energies use the dimer basis, which is the counterpoise
correction. A negative ΔE means that the dimer has a lower energy than the two
separate fragments.

| Layer | Method |
| --- | --- |
| HF | Hartree–Fock |
| MP2 | Second-order Møller–Plesset perturbation theory with a frozen core |
| CASCI | Exact diagonalization in the active space |
| SQD | Sample-based quantum diagonalization in the same active space |
| CCSD(T) | Coupled cluster with a frozen core in the full orbital space |

Each energy has two columns: Ha with six decimals and kcal/mol with three
decimals. A layer without a value shows `—`, and the column `空值原因` (reason
for the empty value) gives the reason. The mode `classical_only` gives only the
HF and MP2 layers.

In the example report, the five layers are between −3.935 and −3.557 kcal/mol.

## Error Decomposition

The table `误差分解` (error decomposition) has four rows. Each row gives the
signed value and the absolute value, in Ha and in kcal/mol.

| Row | Definition | Example report |
| --- | --- | --- |
| `算法误差` (algorithm error) | SQD − CASCI | 0.016 kcal/mol |
| `活性空间误差` (active-space error) | CASCI − same-basis full-space reference | 0.125 kcal/mol |
| `基组误差` (basis error) | Same-basis full-space reference − external reference | Empty |
| `模型误差` (model error) | Active-space error + basis error | Empty |

The line `同基组全空间参考来源：pyscf_ccsd_t` names the source of the
same-basis full-space reference. In the example report, the source is the
CCSD(T) layer of the same run. The basis error and the model error need an
external reference, which the computational side imports. The example run had
no external reference, so the two rows are empty with the reason
`missing external reference`.

## Resource Estimates

The last two tables of each fragment section are planning data, not energies.

- `双因子分解成本估计（QPE 资源）`: a double-factorization cost estimate of
  quantum phase estimation (QPE) for the active-space Hamiltonian of AB, A, and
  B. The columns are λ in Ha, logical qubits, and Toffoli gates. The table is an
  estimate, not the result of a QPE circuit.
- `SQD 工作量与内存预估`: the qubits (two for each active orbital), the full CI
  dimension of the active space, and the predicted peak memory (RSS, in decimal
  MB) of the two SQD stages, prepare and sample.

You can skip these two tables when you compare the energies with experiments.

## Experimental Ks Values

You can give the computational side a table of experimental labels and Ks
values. The computational side passes the table to `render_report` as
`ks_table`. The report then ends with the section `实验 Ks 值（只并列显示）`
(experimental Ks values, listed side by side only). The section has this form:

```text
## 实验 Ks 值（只并列显示）

| 实验标签 | Ks |
|---|---|
| <label> | <Ks> |
```

The rows keep the order in which the computational side passes them. q2m3 does
not compute a correlation, a fit, or a p-value from these values. Any
comparison between ΔE and Ks is your own analysis. The example report has no
such section because its run had no Ks table.
