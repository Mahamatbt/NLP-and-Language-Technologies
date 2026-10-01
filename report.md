# Sequential Models for Vaccine Tweet Sentiment

## Abstract

This study investigates sentiment classification in the Zindi challenge *To Vaccinate or Not to Vaccinate*. Tweets are short, noisy, and often contain hashtags, mentions, URLs, sarcasm, and vaccine-specific language. We audit the provided data, compare two TF-IDF baselines locally, and specify three distinct neural sequence architectures for the full experiment: a BiLSTM, TextCNN, and fine-tuned XLM-R. On one stratified 80/20 validation split, TF-IDF + Logistic Regression obtains 0.726 accuracy and 0.671 macro-F1; TF-IDF + Linear SVM obtains 0.734 accuracy and 0.666 macro-F1. Their negative class F1 scores are 0.507 and 0.481, indicating substantial difficulty with the minority anti-vaccine sentiment class. This environment did not include a neural framework or pretrained model access, so no neural score is claimed. Results therefore constitute a partial experimental comparison; the notebook includes implementations and evaluation cells to complete the comparison in a suitable runtime.

## 1. Problem and data

Vaccine attitudes affect public health communication and uptake. Social media offers timely, informal expression, but automatic sentiment analysis must distinguish support, opposition, and neutral or off-topic discussion. A tweet can mention vaccination without expressing a stance, so lexical topic detection alone is insufficient. Negation, quotation, sarcasm, hashtags, and conversational context can reverse or obscure apparent polarity.

The provided `Train.csv` contains 10,001 physical data rows with fields `tweet_id`, `safe_text`, `label`, and `agreement`. `Test.csv` contains 5,177 tweet IDs and texts; it is unlabeled and is not used for model selection. One training row has a blank target and is excluded. A second malformed row is evidently shifted: its text appears in the ID field, the value `1` appears in `safe_text`, and `2/3` appears in `label`. We recover the text and retain 2/3 as its continuous target. This correction is auditable in the notebook. The resulting 10,000 examples contain 1,038 negative labels, 4,908 neutral labels, 4,053 exact +1 labels, and one fractional target (2/3); rounded classification labels therefore include 4,054 positive examples. Agreement is available but was not used as a sample weight.

Median tweet length is 107 characters (mean 99.9). The labels are imbalanced: negative examples comprise approximately 10.4% of the usable records, while neutral and positive examples comprise approximately 49.1% and 40.5%. Short texts favor sparse n-gram baselines for speed, but informal spelling and hashtags create vocabulary sparsity; sequential encoders may learn compositional signals such as negation and phrase order. The starter notebook shows examples of topic drift and ambiguous mentions, but it is instructional material, not evidence of our model performance.

## 2. Related work and model selection

Sentiment analysis has long used sparse lexical features with linear classifiers as efficient, interpretable baselines (Pang, Lee, and Vaithyanathan, 2002). N-gram TF-IDF preserves local word sequences but does not directly model longer context or word order beyond the selected n-gram window. Logistic Regression provides class probabilities and a strong linear baseline; Linear SVM often performs well in high-dimensional sparse text spaces.

Recurrent neural networks process token sequences in order. LSTMs address vanishing-gradient limitations through gated memory (Hochreiter and Schmidhuber, 1997); bidirectional recurrence uses both left and right context (Schuster and Paliwal, 1997). A BiLSTM is included to test whether learned contextual token representations improve sentiment interpretation. It can be data-hungry and sensitive to vocabulary coverage and regularization.

TextCNN applies convolutional filters over token embeddings to detect local n-gram patterns, followed by max pooling (Kim, 2014). It is a distinct neural sequential architecture and tests whether salient local phrases are sufficient for these short tweets. It is computationally lighter than recurrent or transformer encoders, but fixed-width filters can miss longer dependencies and compositional structure.

Transformers use self-attention to model contextual relationships without recurrent computation (Vaswani et al., 2017). XLM-R was pretrained on multilingual web text and is a reasonable candidate for diverse language variation and transfer (Conneau et al., 2020). The supplied examples are English, but multilingual pretraining may help with informal and cross-lingual lexical variation. Fine-tuning a large model can overfit this modest dataset and has greater compute and deployment cost.

The five approaches are therefore: (1) word TF-IDF + Logistic Regression, (2) word TF-IDF + Linear SVM, (3) BiLSTM with trainable embeddings, (4) TextCNN with trainable embeddings, and (5) fine-tuned XLM-R. The first two establish reproducible lexical baselines; the latter three test recurrent, convolutional, and attention-based sequence learning.

## 3. Experimental methodology

The notebook fixes seed 42 and uses a stratified 80/20 train/validation split, retaining the full usable dataset in the split. The original challenge uses a continuous score and RMSE. The required TF-IDF Logistic Regression and Linear SVM are treated as classifiers: labels are rounded to -1, 0, or +1, and accuracy, macro-F1, per-class precision/recall/F1, and confusion matrices are reported. Separate TF-IDF Ridge and LinearSVR diagnostic regressors provide RMSE/MAE; they are supplementary and not counted among the five assigned approaches. Macro-F1 gives each sentiment class equal weight despite imbalance. No threshold or hyperparameter selection uses the validation labels beyond the fixed baseline comparison.

The TF-IDF baselines use word 1–2 grams, sublinear term frequency, minimum document frequency 2, and at most 200,000 features. Logistic Regression uses balanced class weights; Linear SVM also uses balanced class weights. For both, the notebook reports classification metrics only. Separate Ridge and LinearSVR diagnostic regressors report RMSE/MAE and are not counted as additional approaches in the required five-model comparison. In a complete five-model run, neural model inputs use a fixed token limit and validation-only model selection. XLM-R uses the pretrained tokenizer and sequence classifier head. Report learning curves and per-model confusion matrices after running the neural cells.

### Local baseline results

| Model | Accuracy | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.7260 | 0.6711 | 0.507 | 0.788 | 0.718 |
| TF-IDF + Linear SVM | 0.7335 | 0.6656 | 0.481 | 0.789 | 0.727 |

These classification results are from the same seed-42 stratified split. Separately, continuous-score TF-IDF regressors yield: Ridge RMSE 0.5829 / MAE 0.4353; LinearSVR RMSE 0.6028 / MAE 0.4226. These regressors are diagnostic, not additional approaches in the required five-model comparison.

## 4. Discussion and error analysis

Linear SVM has the higher accuracy by 0.75 percentage points and slightly higher positive-class F1, while Logistic Regression has higher macro-F1 by 0.55 points and better negative-class F1. This illustrates why accuracy alone would conceal a tradeoff affecting the underrepresented negative class. Both classifiers identify neutral tweets well but confuse negative examples with neutral or positive sentiment. The notebook generates the confusion matrices and representative validation errors in `results/` when fully executed in Jupyter.

Likely error mechanisms to verify against those examples include: (a) sarcastic or quoted statements whose literal polarity differs from author stance; (b) negation and long-distance scope (“not saying vaccines are unsafe”); (c) topic-only mentions with no expressed opinion; (d) hashtags and misspellings that split useful terms into rare features; and (e) annotator disagreement. Agreement is 1.0 for 5,866 rows, 2/3 for 3,894, and 1/3 for 239. The lower-agreement subset may contain ambiguous labels, so some apparent model errors may not have a single obvious sentiment. The notebook exports examples with gold labels, predictions, and text so these interpretations can be checked rather than assumed.

No conclusion about neural models is possible until their experiments run. The central research question is thus only partly answered: sparse models provide useful evidence that local lexical cues work for much of the task, while their minority-class errors motivate testing contextual sequence models. The completed experiment should compare validation performance, training curves, confusion matrices, runtime, and error types across all five models.

## 5. Limitations and future work

The reported scores come from a single random split, with no repeated-seed uncertainty estimate. The train/validation examples may share event or vocabulary patterns, and no external temporal or geographic split was available. Label imbalance and disagreement complicate the negative class. We did not use annotator agreement, examine demographic fairness, or evaluate robustness to spelling, hashtags, and distribution shift. Hyperparameter search was intentionally limited. Most importantly, the neural architectures have not been trained in the current environment; this report is not a complete empirical five-model comparison yet.

Next steps are to run the notebook with PyTorch and Transformers, tune only against the fixed validation set (or nested/cross-validation where compute permits), report RMSE/MAE and macro-F1 for all five, and compare per-class errors. Repeated stratified splits or bootstrap intervals would quantify uncertainty. Agreement-weighted training, calibration, class-sensitive losses, character/subword features, and robustness tests for negation, spelling, and code-switching are useful follow-ups. Any final competition submission should be generated only after model selection is complete, with the original test IDs preserved.

## References

- Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale*. ACL. https://aclanthology.org/2020.acl-main.747/
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
- Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification*. EMNLP. https://aclanthology.org/D14-1181/
- Pang, B., Lee, L., & Vaithyanathan, S. (2002). *Thumbs up? Sentiment Classification using Machine Learning Techniques*. EMNLP. https://aclanthology.org/W02-1011/
- Schuster, M., & Paliwal, K. K. (1997). *Bidirectional recurrent neural networks*. IEEE Transactions on Signal Processing, 45(11), 2673–2681. https://doi.org/10.1109/78.650093
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS. https://papers.neurips.cc/paper/7181-attention-is-all-you-need
- Zindi. (n.d.). *To Vaccinate or Not to Vaccinate: It’s not a Question*. Challenge dataset and task description. https://zindi.africa/competitions/to-vaccinate-or-not-to-vaccinate-its-not-a-question
