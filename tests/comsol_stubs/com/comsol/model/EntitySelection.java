package com.comsol.model;

public class EntitySelection {
  public EntitySelection named(String tag) { return this; }
  public EntitySelection set(int... entities) { return this; }
  public EntitySelection add(int... entities) { return this; }
  public EntitySelection all() { return this; }
  public EntitySelection geom(int dim) { return this; }
  public EntitySelection label(String label) { return this; }
}
