# Amazon ML Challenge 2025

Predictive Product Pricing Using BERT

## Overview

This repository contains our solution for the Amazon ML Challenge 2025. The task focuses on estimating product prices based on textual product descriptions and associated attributes. Given the high variability and diversity in product metadata, pricing prediction requires understanding both semantic detail and contextual cues in product text.

## Problem Statement

**Smart Product Pricing Challenge**
Develop a model that predicts the optimal selling price of a product by analyzing its textual attributes. Product pricing is influenced by multiple factors including brand, specifications, quantity, and category positioning. The objective is to model these relationships and produce accurate price estimates for unseen products.

## Dataset

* **Training Set:** 75,000 products with full descriptions and labeled prices
* **Test Set:** 75,000 products used for final evaluation
* **Evaluation Metric:** Symmetric Mean Absolute Percentage Error (SMAPE)

**SMAPE Formula:**

```
SMAPE = (1/n) * Σ |predicted_price - actual_price| / ((|actual_price| + |predicted_price|)/2)
```

## Approach

* Model: **bert-base-uncased**
* Problem treated as a **regression task**
* Product descriptions were tokenized and fed into BERT with a regression head on top.
* Training involved fine-tuning BERT end-to-end on product text and target price values.
* Additional normalization steps applied to price targets to stabilize training.

## Result

Our final submission achieved a **SMAPE score of approximately 45** on the evaluation dataset.
