# CannyEdit: Selective Canny Control and Dual-Prompt Guidance for Training-free Image Editing

**[Project Page](https://vaynexie.github.io/CannyEdit/) | [arXiv (Coming Soon)]() | [Code (Coming Soon)]() | [Demo (Coming Soon)]()**

---

### Authors

Weiyan Xie*, Han Gao*, Didan Deng*, Kaican Li, April Hua Liu, Yongxiang Huang, Nevin L. Zhang

<small><i>*Indicates Equal Contribution</i></small>

### Affiliations

Huawei Hong Kong AI Framework & Data Technologies Lab, The Hong Kong University of Science and Technology, Shanghai University of Finance and Economics

---

## Abstract

Recent advances in text-to-image (T2I) models have enabled training-free regional image editing by leveraging the generative priors of foundation models. However, existing methods struggle to balance text adherence in edited regions, context fidelity in unedited areas, and seamless integration of edits. We introduce **_CannyEdit_**, a novel training-free framework that addresses these challenges through two key innovations: (1) **_Selective Canny Control_**, which masks the structural guidance of Canny ControlNet in user-specified editable regions while strictly preserving the source image’s details in unedited areas via inversion-phase ControlNet information retention. This enables precise, text-driven edits without compromising contextual integrity. (2) **_Dual-Prompt Guidance_**, which combines local prompts for object-specific edits with a global target prompt to maintain coherent scene interactions. On real-world image editing tasks (addition, replacement, removal), CannyEdit outperforms prior methods like KV-Edit, achieving a 2.93%--10.49% improvement in the balance of text adherence and context fidelity. In terms of editing seamlessness, user studies reveal only 49.2% of general users and 42.0% of AIGC experts identified CannyEdit's results as AI-edited when paired with real images without edits, versus 76.08--89.09% for competitor methods.

## 

## ✨ Highlights

### 1. High-Quality Region-Specific Image Edits
Our method enables high-quality region-specific image edits, especially useful in cases where SOTA free-form image editing methods fail to ground edits accurately.

**Prompt:** *Add a cyclist riding the bike, wearing a green and yellow jersey and sunglasses.*

|                            Input                             |                       CannyEdit (ours)                       |                     FLUX.1 Kontext [dev]                     |                            GPT-4o                            |                            Doubao                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/4.jpeg" width="200"> | <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/4_C.jpeg" width="200"> | <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/4_kontext.jpeg" width="200"> | <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/4_gpt.png" width="200"> | <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/4_doubao.png" width="200"> |

---

### 2. Support Multiple Edits at One Pass
Our methods can support edits on multiple user-specific regions at one generation pass when multiple masks are given.

**Prompt:** *Add a slide + Add a group of elderly men and women practicing Tai Chi.*
|                            Input                             |                       CannyEdit (ours)                       |                     FLUX.1 Kontext [dev]                     |                            GPT-4o                            |                            Doubao                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/b1.png" width="200"> | <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/b1_c.png" width="200"> | <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/b1_context.png" width="200"> | <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/b1_4o.png" width="200"> | <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/b1_dou.png" width="200"> |

---

### 3. Precise Local Control

#### a. Control by mask size
By specifying the mask size, our method effectively controls the size of the generated subject.

**Prompt:** *Add a sofa + Add a painting on the wall.*

|                            Input                             |                 CannyEdit <br />(small mask)                 |                CannyEdit <br />(medium mask)                 |                 CannyEdit <br />(large mask)                 |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/33_1.jpg" width="200"> | <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/33_2.jpg" width="200"> | <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/33_3.jpg" width="170"> | <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/33_4.jpg" width="200"> |

#### b. Control by text details
By providing varying local details in the text, subjects with different visual characteristics are generated.

**Prompt:** *Add a woman customer reading a menu + Add a waiter ready to serve.*

|                            Input                             |                     CannyEdit (output 1)                     |                     CannyEdit (output 2)                     |                     CannyEdit (output 3)                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/55_1.png" width="200"> | <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/55_2.png" width="200"> | <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/55_3.png" width="200"> | <img src="https://vaynexie.github.io/CannyEdit/static/figures/highlights/55_4.png" width="200"> |

---

## BibTeX

If you find our work useful, please consider citing:
```bibtex
@article{xie2025canny,
  title={CannyEdit: Selective Canny Control and Dual-Prompt Guidance for Training-free Image Editing},
  author={Xie, Weiyan and Gao, Han and Deng, Didan and Li, Kaican and Liu, April Hua and Huang, Yongxiang and Zhang, Nevin L.},
  journal={arXiv preprint arXiv:xxx},
  year={2025}
}

```
