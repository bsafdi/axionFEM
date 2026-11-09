"""The pinned gmsh coax sliver must raise an actionable error, not abort MFEM."""
import os
import pytest
import axionfem as af
from axionfem.convert import to_3d

@pytest.mark.skipif(not os.environ.get("AXF_SLOW"), reason="3D meshing; set AXF_SLOW=1")
def test_coax_degenerate_element_is_rejected(tmp_path):
    d = af.template('solenoid', R=.3,H=.4,x=.5,tw=.03,tc=.02,g=.02,tp=.03,NI=1e3)
    d.doc['fullwave']={'frequencies':[1e6],'dtn_modes':4}
    d = to_3d(d,mesh={'size':.18,'surface_size':.06,'order':1,'domain_radius':.8})
    with pytest.raises(af.AxionFEMError, match='degenerate 3D mesh element.*coax_short'):
        d.solve(tmp_path/'sliver',verbose=0)
