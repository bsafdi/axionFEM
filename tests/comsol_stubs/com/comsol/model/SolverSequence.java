package com.comsol.model;

public class SolverSequence {
  public SolverSequence study(String tag) { return this; }
  public SolverFeature create(String tag, String type) { return new SolverFeature(); }
  public SolverFeature feature(String tag) { return new SolverFeature(); }
  public FeatureList feature() { return new FeatureList(); }
  public SolverSequence attach(String studyTag) { return this; }
  public void runAll() {}
}
