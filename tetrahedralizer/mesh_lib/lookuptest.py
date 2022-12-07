from tetrahedralizer.DielectricDecomposition.LookupTables.permittivity_lookup import permittivity_lookup

material = "water"
frequencies = 1010000000.0
output = permittivity_lookup(material, frequencies)
print(output)
