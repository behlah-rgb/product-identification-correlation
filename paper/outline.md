# Paper outline - Smart Shelf Market Basket Analysis
**Owner: P4** (each section's technical owner writes the first draft of their part)

Working title: *From Shelf to Basket: Linking Vision-Based Product Detection with Association Rule Mining
and KNN Recommendation*

Target length: 6-8 pages, IEEE conference format (Overleaf → "IEEE Conference Template").

---

## 1. Introduction  *(P4, done - needs edits)*
- Keep the current draft. Replace "Our ipyn files/notebook, for example, uses..." with:
  *"The implementation represents transactions as one-hot vectors for Apriori and uses Jaccard/Hamming
  distance for nearest-neighbour basket search."*
- End with an explicit **contributions** list:
  1. An end-to-end pipeline that converts before/after shelf images into transaction records
  2. A comparison of CNN-based and classical (colour-differencing) change detection on a 3×3 shelf
  3. A controlled study of how detection errors distort discovered association rules (planted-rule recovery)
  4. An evaluation of KNN next-item recommendation on simulated and real (Groceries) baskets

## 2. Related Work  *(P4)*
Three short paragraphs, 3-4 references each:
- **Retail product detection** - SKU-110K, RP2K, HoloSelecta, Locount (Cai et al.), YOLO family
- **Market Basket Analysis** - Agrawal et al. (Apriori), lift/confidence, Groceries dataset studies
- **Recommendation / KNN** - Cover & Hart, item/basket-based collaborative filtering
- Closing sentence: the gap - vision papers stop at detection; MBA papers assume transactions already exist.

## 3. Methodology
3.1 System overview + pipeline figure *(P4 draws it: draw.io / Excalidraw)*
3.2 Image acquisition and preprocessing - ArUco homography, gray-world WB, bilateral filter, CLAHE, grid *(P2)*
3.3 Product detection - YOLOv8n, dataset, training setup *(P1)*
3.4 Change detection and transaction generation - YOLO count difference; classical LAB ΔE + morphology + planogram *(P2)*
3.5 Basket simulation with planted rules and detection-error model *(P3)*
3.6 Association rule mining - Apriori, support/confidence/lift definitions (write the formulas) *(P3)*
3.7 KNN next-item recommendation - Jaccard vs Hamming, leave-one-out Hit@N protocol *(P3)*

## 4. Experimental Setup  *(P1 + P3)*
- Datasets table: Indian Grocery (403 imgs, 10 classes), own shelf photos (N), sessions (N), simulated baskets (2000), Groceries (14,963 baskets, 167 items)
- Hardware: Colab T4, RTX 3050 laptop, Apple M-series - epochs, image size, batch
- Metrics: mAP@50, mAP@50-95, P/R/F1, rule recall & precision@k, Hit@1/Hit@3

## 5. Results  *(fill from `IVP_Project_data/results/`)*

**Table 1 - Detection (Exp 1)** · `yolo_public_only.json`, `exp1_domain_gap.csv`
| Training data | Test set | mAP@50 | mAP@50-95 | P | R |
|---|---|---|---|---|---|
| Public only | Public test | | | | |
| Public only | Shelf test | | | | |
| Public + shelf | Shelf test | | | | |

**Table 2 - Preprocessing ablation (Exp 2)** · `ablation_preprocessing.csv`
| Condition | mAP@50 raw | mAP@50 + preprocessing |
|---|---|---|
| clean / dim / uneven / noise / combined | | |

**Table 3 - Change detection (Exp 3)** · `exp3_change_detection.csv`
| Method | Precision | Recall | F1 |
|---|---|---|---|
| Classical (LAB ΔE + planogram) | | | |
| YOLOv8 count difference | | | |

**Figure - Noise sweep (Exp 4)** · `exp4_noise_sweep.png` - *key result linking IVP and AI*
**Table 4 - Rule recovery** · clean vs through-detector (notebook 04 §3)

**Table 5 - KNN (Exp 5, 5b)** · `exp5_knn.csv`, `exp5b_knn_groceries.csv`
| Data | Popularity Hit@3 | KNN-Hamming Hit@3 | KNN-Jaccard Hit@3 |
|---|---|---|---|
| Simulated | | | |
| Groceries (real) | | | |

**Table 6 - Top "surprising" rules on Groceries** · notebook 05

## 6. Discussion  *(all)*
- Why Jaccard > Hamming on sparse baskets (Hamming rewards shared absences)
- Detection-accuracy threshold below which correlations disappear (from Exp 4)
- Classical method needs a planogram; YOLO doesn't - but classical needs no training data
- Limitations: simulated transactions, small shelf, controlled lighting, one product per cell, small real baskets in Groceries

## 7. Conclusion and Future Work  *(P4)*
Live video instead of before/after stills; hand/occlusion handling; larger shelves; real store POS data.

---

## References (verify every one on Google Scholar before citing - P4)
1. R. Agrawal, T. Imieliński, A. Swami, "Mining association rules between sets of items in large databases," *ACM SIGMOD*, 1993.
2. R. Agrawal, R. Srikant, "Fast algorithms for mining association rules," *VLDB*, 1994.
3. J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You Only Look Once: Unified, real-time object detection," *CVPR*, 2016.
4. G. Jocher, A. Chaurasia, J. Qiu, "Ultralytics YOLOv8," software, 2023. https://github.com/ultralytics/ultralytics
5. E. Goldman et al., "Precise detection in densely packed scenes," *CVPR*, 2019. (SKU-110K)
6. J. Peng et al., "RP2K: A large-scale retail product dataset for fine-grained image classification," arXiv:2006.12634, 2020.
7. K. Fuchs et al., "HoloSelecta dataset: 10'035 GTIN-labelled product instances in vending machines for object detection of packaged products in retail environments," *Data in Brief*, 2020.
8. Y. Cai et al., "Rethinking object detection in retail stores," *AAAI*, 2021. arXiv:2003.08230
9. K. Zuiderveld, "Contrast limited adaptive histogram equalization," *Graphics Gems IV*, 1994.
10. C. Tomasi, R. Manduchi, "Bilateral filtering for gray and color images," *ICCV*, 1998.
11. G. Buchsbaum, "A spatial processor model for object colour perception," *J. Franklin Institute*, 1980. (gray-world)
12. S. Garrido-Jurado et al., "Automatic generation and detection of highly reliable fiducial markers under occlusion," *Pattern Recognition*, 2014. (ArUco)
13. Z. Wang, A. Bovik, H. Sheikh, E. Simoncelli, "Image quality assessment: from error visibility to structural similarity," *IEEE TIP*, 2004.
14. T. Cover, P. Hart, "Nearest neighbor pattern classification," *IEEE Trans. Information Theory*, 1967.
15. B. Sarwar et al., "Item-based collaborative filtering recommendation algorithms," *WWW*, 2001.
16. AgentSK47, "Indian Grocery Object Detection dataset," Roboflow Universe. https://universe.roboflow.com/agentsk47/indian-grocery-object-detection-mfsnx
17. H. Dedhia, "Groceries dataset," Kaggle, 2020. https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset
