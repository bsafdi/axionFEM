#include "../src/shell_map.hpp"
#include <mfem.hpp>
#include <nlohmann/json.hpp>
#include <cstdio>
#include <iostream>

int main(int argc, char** argv) {
  if (argc > 1) {
    nlohmann::json rows = nlohmann::json::array();
    for (int i=1;i<argc;++i) {
      mfem::Mesh mesh(argv[i],1,1,true);
      double worst=0; int index=-1, rejected=0;
      for (int e=0;e<mesh.GetNE();++e) {
        double low=mesh.GetElementSize(e,1), high=mesh.GetElementSize(e,2);
        if (!axf::detail::valid_element_width(low,high)) ++rejected;
        double ratio=low>0 ? high/low : 1e300;
        if (ratio>worst) {worst=ratio;index=e;}
      }
      rows.push_back({{"mesh",argv[i]},{"elements",mesh.GetNE()}, {"max_jacobian_aspect",worst},
                      {"worst_element",index}, {"rejected",rejected}});
    }
    std::cout << rows.dump(2) << std::endl;
    return 0;
  }
  // Real tetrahedra on both sides of the aspect cutoff. MFEM measures singular
  // values of the Jacobian relative to a perfect tetrahedron, not edge lengths.
  for (double height : {4e-6, 5e-7, 1e-13}) {
    mfem::Mesh mesh(3,4,1,0,3);
    mesh.AddVertex(0,0,0); mesh.AddVertex(1,0,0); mesh.AddVertex(0,1,0); mesh.AddVertex(0,0,height);
    mesh.AddTet(0,1,2,3); mesh.FinalizeTetMesh();
    double low=mesh.GetElementSize(0,1), high=mesh.GetElementSize(0,2);
    bool expected = height == 4e-6;
    if (axf::detail::valid_element_width(low,high) != expected) return 1;
    if (expected && !(high/low > 1e5 && high/low < 1e6)) return 2;
  }
  std::puts("Jacobian aspect cutoff on actual near-flat tetrahedra: passed");
}
