"""
This file imports a Jupyter notebook and then removes all of the cells
except markdown cells.  A new Jupyter notebook is then created out of the
markdown cells and saved to a file called report_markdown_only.ipynb (or
whatever string is contained in the variable markdown_only_file).

The number of words in the markdown cells are also counted.
"""

import nbformat as nbf

# File name of Jupyter notebook for the report
report_file = "report.ipynb"

# Name of the new markdown-only Jupyter notebook that will be saved
markdown_only_file = "report_markdown_only.ipynb"

# import report
ntbk = nbf.read(report_file, nbf.NO_CONVERT)

# initialise work count and list of markdown cells (these cells will be used to
# create the new notebook)
wordCount = 0
cells_to_keep = []

# loop over all of the cells in the original Jupyter notebook
for cell in ntbk.cells:

    # check whether the current cell is a markdown cell
    if cell.cell_type == "markdown":

        # add the markdown cell to the list of cells
        cells_to_keep.append(cell)

        # counting words
        content = cell['source']
        new_words = len(content.split())
        wordCount += new_words


print("The markdown cells contain", wordCount, "words")


# create a new Jupyter notebook. first make sure the file names are
# different to avoid overwriting the original
if report_file == markdown_only_file:
    print('File name of original report and new report are the same')
    print('Script terminating to avoid overwriting original file')
else:
    new_ntbk = ntbk
    new_ntbk.cells = cells_to_keep
    nbf.write(new_ntbk, markdown_only_file, version=nbf.NO_CONVERT)
