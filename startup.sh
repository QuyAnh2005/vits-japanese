cd monotonic_align
python setup.py build_ext --inplace
cd ..

curl -L https://www.dropbox.com/s/e0h13tufx2oobn2/G_523000.pth?dl=0 --output logs/jp_base/model.pth