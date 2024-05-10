#!/snap/bin/pyroot

from array import array
import ROOT

myFile = ROOT.TFile.Open("file.root", "RECREATE")
tree = ROOT.TTree("tree", "The Tree Title")

# Provide a one-element array, so ROOT can read data from this memory.
var1=array('f',[0])
var2=array('f',[0])
tree.Branch("var1", var1, "var1/F");
tree.Branch("var2", var2, "var2/F");

for i in range(1000):
   var1[0]=0.3*i
   var2[0]=0.5*i
   # Fill the current value of `var` into `branch0`
   tree.Fill()

# Now write the header
tree.Write()
