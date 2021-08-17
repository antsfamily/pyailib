
import pyailib as pl

path = '../../data/files/english_sentence.txt'

h = pl.HuffmanCoding(path)

output_path = h.compress()
print("Compressed file path: " + output_path)

decom_path = h.decompress(output_path)
print("Decompressed file path: " + decom_path)
