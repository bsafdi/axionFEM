package com.comsol.model;

public class PhysicsFeature {
  public PhysicsFeature set(String p, String v) { return this; }
  public PhysicsFeature set(String p, String[] v) { return this; }
  public PhysicsFeature set(String p, String[][] v) { return this; }
  public PhysicsFeature set(String p, double v) { return this; }
  public PhysicsFeature set(String p, int v) { return this; }
  public PhysicsFeature set(String p, int[] v) { return this; }
  public PhysicsFeature set(String p, boolean v) { return this; }
  public PhysicsFeature label(String label) { return this; }
  public PhysicsFeature active(boolean on) { return this; }
  public EntitySelection selection() { return new EntitySelection(); }
}
