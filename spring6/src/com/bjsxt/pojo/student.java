package com.bjsxt.pojo;

import java.util.List;

import org.springframework.beans.factory.annotation.Value;

public class student {
	private int id;
	@Value("${position}")
	private String name;
	private Teacher t;
	public student() {
	}
	public student(int id, String name, Teacher t) {
		this.id = id;
		this.name = name;
		this.t = t;
	}
	public student(Teacher t) {
		this.t = t;
	}
	public Teacher getT() {
		return t;
	}
	public void setT(Teacher t) {
		this.t = t;
	}
	
	public int getId() {
		return id;
	}
	public void setId(int id) {
		this.id = id;
	}
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	@Override
	public String toString() {
		return "student [id=" + id + ", name=" + name + ", t=" + t + "]";
	}
	
}
