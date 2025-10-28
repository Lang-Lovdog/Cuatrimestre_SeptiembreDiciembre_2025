#!/bin/bash

pandoc -H Notes.preamble.tex --pdf-engine=xelatex -o Notes.pdf Notes.md
