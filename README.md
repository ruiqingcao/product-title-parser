## Project Metadata

<table>
  <tr>
    <td><strong>Title</strong></td>
    <td>product-title-parser</td>
  </tr>
  <tr>
    <td><strong>License</strong></td>
    <td>Apache 2.0</td>
  </tr>
  <tr>
    <td><strong>Short Description</strong></td>
    <td>A stripped-down version of the original KeyIntentNER-T project. Only extract Google topics (i.e., standardised product categories based on the Google Product Taxonomy) from short product titles or product descriptions</td>
  </tr>
</table>

---
# product-title-parser
product-title-parser is a tool for researchers to quickly classify product keywords, titles, and descriptions into standardised categories widely used for cataloging e-commerce entries. It uses modern NLP methods to extract and standardise information from short text strings.

## Features

### Standardised product categories
Matches short text strings to Google's well-known Content and Product taxonomies.

## Usage
- Create a virtual environment and pip install the packages in requirements.txt
- ``` python standardise_google.py [inputfilename] [outputfilename] ```
- ```[inputfilename]``` contains a list of short text strings (one per line, up to 100 MAX)
- Results are saved as a CSV file in ```[outputfilename]```
- Data processing can take anywhere from 30 seconds up to ~2 minutes

Example text strings (see example.txt): 
```
Wireless Noise-Cancelling Earbuds
Reusable Stainless-Steel Water Bottle 1 L
Foldable Laptop Stand with Cooling Vent
Memory-Foam Seat Cushion for Office Chairs
Quick-Charge USB-C Wall Adapter 65 W
Non-Stick Ceramic Frying Pan 28 cm
Compact Air Purifier with HEPA Filter
Solar-Powered Outdoor String Lights 10 m
Adjustable Resistance Bands Set (5 Levels)
Waterproof Portable Bluetooth Speaker
```

### Notes on Data:
- Standardised product categories come from [Google Product Taxonomy](https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt)
- A smaller sentence transformers model is used which does not perform as well with some of the Product categories. According to the original KeyIntentNER-T project, the [all-roberta-large-v1 model](https://huggingface.co/sentence-transformers/all-roberta-large-v1) performed best for sample keywords in testing. 
  
