package com.comsol.model;

public class Study {
  public StudyFeature create(String tag, String type) { return new StudyFeature(); }
  public StudyFeature feature(String tag) { return new StudyFeature(); }
  public Study label(String label) { return this; }
}
