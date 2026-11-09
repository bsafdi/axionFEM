package com.comsol.model;

public class GeomSequence {
  public GeomSequence axisymmetric(boolean on) { return this; }
  public GeomSequence lengthUnit(String unit) { return this; }
  public GeomFeature create(String tag, String type) { return new GeomFeature(); }
  public GeomFeature feature(String tag) { return new GeomFeature(); }
  public GeomSelectionList selection() { return new GeomSelectionList(); }
  public GeomSelection selection(String tag) { return new GeomSelection(); }
  public void run() {}
  public void run(String tag) {}
}
