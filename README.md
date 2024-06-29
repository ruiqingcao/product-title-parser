# KeyIntentNER-T
KeyIntentNER-T is a tool designed for SEO and digital marketing professionals to provide valuable keyword insights. By entering a list of keywords, you can gain insights into Keyword Intent, NLP Entities extracted via NER (Named Entity Recognition), and Topics. This tool demonstrates how modern NLP methods can be used to understand shorter text strings (keywords) in a way similar to search engines.

## Features
### Keyword Intent
A custom function that looks for the presence of specific terms in keywords and classifies them into one of six predefined intent categories:

- Informational
- Navigational
- Local
- Commercial Investigation
- Transactional
- Other

### NLP Entities
Utilizes GLiNER, an advanced Named Entity Recognition (NER) model, to classify shorter text strings. Entities are mapped to all entity types included in the Google Cloud Natural Language API.

### Topics
Matches keywords to topics from Google's well-known Content and Product taxonomies.

## Usage
Enter a list of keywords (one per line, up to 100 MAX) and click the submit button. Keyword processing can take anywhere from 30 seconds up to ~2 minutes due to the extensive analysis performed behind the scenes. Once processing is complete, you can download any of the bar chart images and download a CSV export with insights for all keywords.

## Benefits for SEO
Improved Content Strategy
Focus your SEO efforts on creating more relevant and helpful content that addresses the search intent for keywords.

### Enhanced Keyword Targeting
Match keywords to Google's well-known categories, ensuring your content is aligned with popular search themes.

### Better Understanding of User Intent
Gain insights into what kind of information a person is looking for and how keywords can be interpreted by search engines.

#### GLiNER Model Citation
- GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer.
- Authors: Urchade Zaratiana, Nadi Tomeh, Pierre Holat, Thierry Charnois.
- Year: 2023.
- Link: [arXiv:2311.08526](https://arxiv.org/abs/2311.08526)

For questions or if you are interested in building custom SEO dash apps, contact me at: jrad.seo@gmail.com
