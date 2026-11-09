package com.comsol.model;

public class StudyFeature {
  public StudyFeature set(String p, String v) { return this; }
  public StudyFeature set(String p, String[] v) { return this; }
  public StudyFeature set(String p, String[][] v) { return this; }
  public StudyFeature set(String p, double v) { return this; }
  public StudyFeature set(String p, int v) { return this; }
  public StudyFeature set(String p, int[] v) { return this; }
  public StudyFeature set(String p, boolean v) { return this; }
  public StudyFeature label(String label) { return this; }
  public StudyFeature activate(String physicsTag, boolean on) { return this; }
}
