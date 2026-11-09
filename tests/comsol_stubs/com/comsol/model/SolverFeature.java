package com.comsol.model;

public class SolverFeature {
  public SolverFeature set(String p, String v) { return this; }
  public SolverFeature set(String p, String[] v) { return this; }
  public SolverFeature set(String p, String[][] v) { return this; }
  public SolverFeature set(String p, double v) { return this; }
  public SolverFeature set(String p, int v) { return this; }
  public SolverFeature set(String p, int[] v) { return this; }
  public SolverFeature set(String p, boolean v) { return this; }
  public SolverFeature label(String label) { return this; }
  public SolverFeature create(String tag, String type) { return new SolverFeature(); }
  public SolverFeature feature(String tag) { return new SolverFeature(); }
  public FeatureList feature() { return new FeatureList(); }
}
