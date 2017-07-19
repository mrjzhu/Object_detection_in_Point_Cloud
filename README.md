# Object detection in Point Cloud: lane marking

## Description

Object detection in Point Cloud is popular in HD Map and sensor-based autonomous driving. There basically four types of object we can obtain in daily scenario: 
road surface - contains painted lane marking and pavement area, 
support facility - contains road boundary (guardrail and curb), road sign, light pole, etc., uncorrelated object - for example, sidewalk, building, etc.
moving object - such like pedestrian, vehicle, bicycle, etc.

## Introduction

In this project, we design and prototype our lane marking detection algorithm to mark the lane.

## Steps

1. Filter points about latitude and longitude with intensity
2. Group the result from last step into three lanes with the k-means++ algorithm in a 3D coordinate system.
3. Fit lines and exclude the abnormal points with the RANSAC algorithm
4. mark the result.

## Notes
See the detailed process in file lane mark.ipynb
