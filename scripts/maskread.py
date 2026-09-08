import os, mmap, struct, sys
# usage: maskread.py <bdf> <addr> [addr...]
# prints one 0x%08x value per address, space separated
bdf = sys.argv[1]
addrs = [int(a, 0) for a in sys.argv[2:]]
fd = os.open(f"/sys/bus/pci/devices/{bdf}/resource0", os.O_RDONLY)
m = mmap.mmap(fd, 16 << 20, mmap.MAP_SHARED, mmap.PROT_READ)
print(" ".join("0x%08x" % struct.unpack_from("<I", m, a)[0] for a in addrs))
os.close(fd)
