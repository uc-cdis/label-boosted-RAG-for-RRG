# Metrics

This folder contains three metric repositories modified for **LaB-RAG**:

- **CLEAR** → `CLEAR-evaluator`
  Modified: https://github.com/tnguyen2907/CLEAR-evaluator
  Original: https://github.com/ChicagoHAI/CLEAR-evaluator

- **GREEN** → `GREEN`  
  Modified: https://github.com/tnguyen2907-AIMI/GREEN
  Original: https://github.com/Stanford-AIMI/GREEN

- **RadCliQ** → `CXR-Report-Metric`  
  Modified: https://github.com/tnguyen2907/CXR-Report-Metric
  Original: https://github.com/Stanford-AIMI/GREEN

- **RaTEScore** → `RaTEScore`  
  Modified: https://github.com/tnguyen2907/RaTEScore
  Original: https://github.com/MAGIC-AI4Med/RaTEScore

These are **forked and modified** versions of the original repositories to support **LaB-RAG**. If you need to change a metric, make the edits inside the corresponding repository in this folder, then reinstall it there, for example:

```bash
cd metrics/<repo-name>
pip install -e .