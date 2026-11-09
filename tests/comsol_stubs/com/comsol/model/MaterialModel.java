package com.comsol.model;

public class MaterialModel {
  public MaterialModel set(String p, String v) { return this; }
  public MaterialModel set(String p, String[] v) { return this; }
  public MaterialModel set(String p, String[][] v) { return this; }
  public MaterialModel set(String p, double v) { return this; }
  public MaterialModel set(String p, int v) { return this; }
  public MaterialModel set(String p, int[] v) { return this; }
  public MaterialModel set(String p, boolean v) { return this; }
  public MaterialModel label(String label) { return this; }
  public MaterialModel addInput(String quantity) { return this; }
}
