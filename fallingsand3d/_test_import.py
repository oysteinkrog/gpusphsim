import sys
sys.path.insert(0, r'C:\WORK\gpusphsim\fallingsand3d')
try:
    import world
    print("world OK")
except Exception as e:
    print(f"world FAIL: {e}")
try:
    import counting_sort
    print("counting_sort OK")
except Exception as e:
    print(f"counting_sort FAIL: {e}")
