package com.comsol.model;

public class ResultFeature {
  public ResultFeature set(String p, String v) { return this; }
  public ResultFeature set(String p, String[] v) { return this; }
  public ResultFeature set(String p, String[][] v) { return this; }
  public ResultFeature set(String p, double v) { return this; }
  public ResultFeature set(String p, int v) { return this; }
  public ResultFeature set(String p, int[] v) { return this; }
  public ResultFeature set(String p, boolean v) { return this; }
  public ResultFeature label(String label) { return this; }
  public ResultFeature setIndex(String p, String v, int index) { return this; }
}
