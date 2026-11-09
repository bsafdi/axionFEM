package com.comsol.model;

public class EvaluationGroup {
  public EvaluationGroup set(String p, String v) { return this; }
  public EvaluationGroup set(String p, String[] v) { return this; }
  public EvaluationGroup set(String p, String[][] v) { return this; }
  public EvaluationGroup set(String p, double v) { return this; }
  public EvaluationGroup set(String p, int v) { return this; }
  public EvaluationGroup set(String p, int[] v) { return this; }
  public EvaluationGroup set(String p, boolean v) { return this; }
  public EvaluationGroup label(String label) { return this; }
  public ResultFeature create(String tag, String type) { return new ResultFeature(); }
  public ResultFeature feature(String tag) { return new ResultFeature(); }
  public void run() {}
  public void save(String filename) {}
}
