for i in {01..30}; do python3 ocr_mdx_pipeline.py  pdfs/en_01_Majmoo_alFatawa_IbnBaz.pdf 01  2>&1 | tee -a output-01.log; done
