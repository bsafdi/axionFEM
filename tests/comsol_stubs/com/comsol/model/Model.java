package com.comsol.model;

/** Compile-time stand-in for com.comsol.model.Model (see tests/comsol_stubs/README.md). */
public class Model {
  public ModelParam param() { return new ModelParam(); }
  public ModelNodeList component() { return new ModelNodeList(); }
  public ModelNode component(String tag) { return new ModelNode(); }
  public Model label(String label) { return this; }
  public StudyList study() { return new StudyList(); }
  public Study study(String tag) { return new Study(); }
  public SolverSequenceList sol() { return new SolverSequenceList(); }
  public SolverSequence sol(String tag) { return new SolverSequence(); }
  public Results result() { return new Results(); }
  public void save(String path) throws java.io.IOException {}
}
