# **Change Log** 📜📝

All notable changes to the "**Intent Classification Service**" WhatItIs/program/extension/API/whatever will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---
## [v1.0.0] - 2023-10-21

### Added

* Added BERT model - `bert-base-multilingual-cased` from huggingface as multilingual encoder model
* Added error handling codes for /intent endpoint as per the following specifications:
  * BODY_MISSING, 400
  * TEXT_MISSING, 400
  * INVALID_TYPE, 400
  * TEXT_EMPTY, 400
  * INTERNAL_ERROR, 500
* Added model training and production evaluation notebook in `notebooks/`
* Added Docker files
* Added POSTMAN api documentation
### Changed

* Modifed the /ready logic with simple `model` is `None` check as the `IntentClassifier` does not have any `ready()` method.

### Fixed

* In the initial code the model was loaded after flask server was up. Moved the model loading before starting server fixed the issue.
