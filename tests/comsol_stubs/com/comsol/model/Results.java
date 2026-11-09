package com.comsol.model;

public class Results {
  public EvaluationGroupList evaluationGroup() { return new EvaluationGroupList(); }
  public EvaluationGroup evaluationGroup(String tag) { return new EvaluationGroup(); }
}
