#!/usr/bin/env python3
"""Build the npj Digital Medicine version of the manuscript.

Nature Portfolio structure differs from the current draft in ways that are structural,
not cosmetic:
  - Methods move to the END, after Discussion
  - section numbering is dropped (Nature style uses unnumbered headings)
  - the abstract is a single unstructured paragraph (~200 words), not Aims/Methods/Conclusion
  - Data availability / Code availability are their own top-level sections
  - Author contributions and Competing interests are required sections

The source manuscript is not modified; this writes a separate file so the EHJ-DH
version stays intact if we fall back to it.
"""
import io, re, os

SUB = (r"d:/桌面/_按项目整理/02_生信数据与医学AI项目/07_医学AI_深度学习_多项目孵化/"
       r"医学AI深度学习_导师课题项目群/崔老师项目_PETCT多任务_空间转录组深度学习/"
       r"PETCT多任务/论文_AIECG")
SRC = f"{SUB}/manuscript.md"
DST_DIR = f"{SUB}/submission_npjDigitalMedicine"
os.makedirs(DST_DIR, exist_ok=True)

t = io.open(SRC, encoding="utf-8").read()

def grab(start_marker, end_marker):
    a = t.index(start_marker)
    b = t.index(end_marker)
    return t[a:b].rstrip()

title = t[:t.index("---")].strip()
intro    = grab("## 1. Introduction", "## 2. Methods")
methods  = grab("## 2. Methods", "## 3. Results")
results  = grab("## 3. Results", "## 4. Discussion")
discuss  = grab("## 4. Discussion", "## 6. Figure legends")
figleg   = grab("## 6. Figure legends", "## 5. Declarations")
decl     = grab("## 5. Declarations", "## References")
refs     = t[t.index("## References"):].rstrip()

# strip section numbering, Nature style
def unnumber(s):
    s = re.sub(r"^## \d+\.\s*", "## ", s, flags=re.M)
    s = re.sub(r"^### \d+\.\d+(\.\d+)?\s*", "### ", s, flags=re.M)
    return s

intro, methods, results, discuss, figleg = map(unnumber, (intro, methods, results, discuss, figleg))

ABSTRACT = """## Abstract

AI-enabled electrocardiography can estimate concurrent echocardiographic findings, but prior work frames the electrocardiogram as a surrogate for the echocardiogram being performed at the same moment. We asked whether the AI-ECG structural signature carries prognostic information *beyond* that echocardiogram. Using MIMIC-IV, MIMIC-IV-ECG and MIMIC-IV-ECHO, we paired each 12-lead ECG to the nearest transthoracic echocardiogram within ±7 days (42,059 patients; patient-level splits; test n=8,646). BEACON-ECG, a validation-selected ensemble of multitask convolutional networks, achieved test AUROCs of 0.900 for HFrEF, 0.860 for severe aortic stenosis, 0.789 for left-ventricular hypertrophy and 0.749 for 1-year mortality, exceeding a tabular ECG machine-measurement comparator on identical patients (DeLong *P*<0.001 for all). Performance was preserved on an independent health system without retraining (EchoNext, n=4,827; HFrEF 0.896). Among patients whose echocardiogram was normal (LVEF>50%, n=6,124), the highest quintile of AI-ECG structural risk had higher 1-year all-cause mortality (24.0% vs 13.5%; adjusted hazard ratio 1.85, 95% CI 1.61–2.13). Serial-ECG trajectories added nothing over a single tracing at the deep-representation level. Performance was not uniform across subgroups: hypertrophy risk was under-predicted in Black patients independently of age (adjusted odds ratio 1.66, 95% CI 1.28–2.15). Prospective evaluation, and correction of this disparity, are required before clinical use."""

# pull the availability statements out of Declarations into their own sections
def extract(label, block):
    m = re.search(rf"\*\*{label}\.?\*\*\s*(.+?)(?=\n\n\*\*|\Z)", block, re.S)
    return m.group(1).strip() if m else None

data_av = extract("Data availability", decl)
code_av = extract("Code availability", decl)
ethics  = extract("Ethics", decl)
protreg = extract("Protocol and registration", decl)

NPJ = f"""{title}

---

{ABSTRACT}

---

{intro}

{results}

{discuss}

{methods}

### Ethics

{ethics}

### Reporting

This study is reported in accordance with **TRIPOD+AI**; the completed checklist is provided as a supplementary file. {protreg}

## Data availability

{data_av}

## Code availability

{code_av}

{refs}

## Acknowledgements

*(To be completed — funding sources, and any individuals who contributed but do not meet authorship criteria.)*

## Author contributions

*(To be completed. npj Digital Medicine requires a statement specifying each author's contribution — e.g. conceptualisation, data curation, formal analysis, methodology, software, writing — original draft, writing — review and editing. With a single author this reads: Z.T. conceived the study, performed all analyses, and wrote the manuscript.)*

## Competing interests

*(To be completed. If none: "The author declares no competing interests." Nature Portfolio requires this statement even when there are no competing interests.)*

{figleg}
"""

io.open(f"{DST_DIR}/manuscript_npjDM.md", "w", encoding="utf-8").write(NPJ)

# word counts
body = NPJ[NPJ.index("## Introduction"):NPJ.index("## Data availability")]
main_no_methods = NPJ[NPJ.index("## Introduction"):NPJ.index("## Methods")]
abst = re.search(r"## Abstract\n\n(.+?)\n\n---", NPJ, re.S).group(1)
print(f"abstract          : {len(abst.split()):>5} words  (npj target ~200, unstructured)")
print(f"Intro+Results+Disc: {len(main_no_methods.split()):>5} words")
print(f"incl. Methods     : {len(body.split()):>5} words")
print(f"\nwritten: {DST_DIR}/manuscript_npjDM.md")
print("\nsection order:")
for m in re.finditer(r"^## (.+)$", NPJ, re.M):
    print("  ", m.group(1))
