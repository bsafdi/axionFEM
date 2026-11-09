package com.comsol.model;

public class ModelNode {
  public GeomList geom() { return new GeomList(); }
  public GeomSequence geom(String tag) { return new GeomSequence(); }
  public MeshList mesh() { return new MeshList(); }
  public MeshSequence mesh(String tag) { return new MeshSequence(); }
  public PhysicsList physics() { return new PhysicsList(); }
  public Physics physics(String tag) { return new Physics(); }
  public MaterialList material() { return new MaterialList(); }
  public Material material(String tag) { return new Material(); }
  public SelectionList selection() { return new SelectionList(); }
  public EntitySelection selection(String tag) { return new EntitySelection(); }
}
