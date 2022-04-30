HOME=${pwd}

#cd $HOME/src/cuda
echo "Installing cuda extension..."
python setup.py clean && python setup.py install