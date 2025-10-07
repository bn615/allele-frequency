import numpy as np
from typing import Tuple


Haplotype = Tuple[int, int, int]
Genotype = Tuple[Haplotype, Haplotype]
Phenotype = Tuple[int, int, int, int, int, int]

def generate_haplotypes():
    # From Navajo Indian data set (n= 38)
    # Returns list of phenotypes, number of alleles, and a map of alleles to numeric code

