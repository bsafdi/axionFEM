package com.comsol.model;

public class Physics {
  public EntitySelection selection() { return new EntitySelection(); }
  public PhysicsFeature create(String tag, String type, int sdim) { return new PhysicsFeature(); }
  public PhysicsFeature feature(String tag) { return new PhysicsFeature(); }
}
