# NLP_Project
HSLU NLP Project

**Q3 (MLM pretraining → fine-tune) foundation is in place** on `ft-qn-3`: `mlm_pretrain.py` (HF Trainer wrapper around `BertForMaskedLM`, MPS-safe), single notebook `q3_mlm_then_finetune.ipynb` scaffolded with full step-by-step documentation, 60+ tests green. See `docs/superpowers/plans/2026-05-18-q3-mlm-pretrain.md` for the plan and `CLAUDE.md` § "Q3 status" for the run-the-notebook recipe.

**Q2 (BERT fine-tuning) foundation is in place** on `ft-qn-2`: `bert_data.py` + `bert_train.py` (HF Trainer wrapper, MPS-safe), notebooks `q2a`–`q2d` scaffolded, 55+ tests green. See `docs/superpowers/plans/2026-05-15-q2-bert-finetune.md` for the plan and `CLAUDE.md` § "Q2 status" for the run-the-notebooks recipe.

Q1 foundation is complete. Branch ft-qn-1 now has 18 commits and a 37-test pytest suite that passes end-to-end.
                                                                                                                              
  What's in place:                      
  - src/nlp_project/ — full package (data loading + preprocessing + train/val split, word2vec + mean/mean+max pooling, TF-IDF,
   MLP, training loop with early stopping + W&B hook, eval + confusion matrix, t-SNE viz)                                     
  - tests/ — 37 tests, all green; slow tests gated with -m slow                                                               
  - scripts/setup_nltk.py — one-time NLTK download                                                                            
  - notebooks/q1a–q1d.ipynb — drafted, not yet executed                                                                       
  - docs/superpowers/{specs,plans} — design and implementation plan committed                                                 
  - CLAUDE.md updated with the Q1 workflow and a W&B URL placeholder                                                          
                                                                                                                              
  What you still need to run locally:                                                                                         
  1. wandb login (one-time)                                                                                                   
  2. uv run jupyter lab → run Q1a → Q1b → Q1c → Q1d                                                                           
  3. In Q1d's last cell, paste the accuracy/F1 numbers from Q1b and Q1c output, then re-run that cell to produce
  figures/metric_comparison_table.csv                                                                                         
  4. Paste the W&B project URL into the placeholder line at the bottom of CLAUDE.md                                           
  5. Commit the resulting notebook outputs and figures (figures are gitignored except .gitkeep; the four expected ones are    
  listed in the spec under §7)                                                                                                
                                                                                                                              
  The notebooks include each cell's purpose as a comment, so the lecturer reading the notebooks gets the narrative without you
   having to re-explain it in the report.                                                                                     
                                                            
✻ Brewed for 4m 0s                                                                                                            
                                                                                          
※ recap: Goal: build Q1 of the HSLU NALAPRO project (two-layer MLP on 20 Newsgroups). The reusable package, 37 passing tests, 
  and four draft notebooks are committed on ft-qn-1. Next: run `wandb login`, then execute the Q1a–Q1d notebooks in Jupyter.  
  (disable recaps in /config)                  