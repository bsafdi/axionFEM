package com.comsol.model;

public class GeomFeature {
  public GeomFeature set(String p, String v) { return this; }
  public GeomFeature set(String p, String[] v) { return this; }
  public GeomFeature set(String p, String[][] v) { return this; }
  public GeomFeature set(String p, double v) { return this; }
  public GeomFeature set(String p, double[] v) { return this; }
  public GeomFeature set(String p, double[][] v) { return this; }
  public GeomFeature set(String p, int v) { return this; }
  public GeomFeature set(String p, int[] v) { return this; }
  public GeomFeature set(String p, boolean v) { return this; }
  public GeomFeature label(String label) { return this; }
  public GeomObjectSelection selection(String name) { return new GeomObjectSelection(); }
  public GeomSequence geom() { return new GeomSequence(); }
}
