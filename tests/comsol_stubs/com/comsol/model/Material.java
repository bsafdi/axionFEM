package com.comsol.model;

public class Material {
  public Material set(String p, String v) { return this; }
  public Material set(String p, String[] v) { return this; }
  public Material set(String p, String[][] v) { return this; }
  public Material set(String p, double v) { return this; }
  public Material set(String p, int v) { return this; }
  public Material set(String p, int[] v) { return this; }
  public Material set(String p, boolean v) { return this; }
  public Material label(String label) { return this; }
  public MaterialModel propertyGroup(String tag) { return new MaterialModel(); }
  public EntitySelection selection() { return new EntitySelection(); }
}
