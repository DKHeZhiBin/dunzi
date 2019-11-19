package com.bjsxt.pojo;


import org.aspectj.lang.annotation.Pointcut;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;
import org.springframework.stereotype.Component;


public class test {
	public static void main(String[] args) {
		ApplicationContext ac = new ClassPathXmlApplicationContext("applicationContext.xml");
		student s = ac.getBean("s",student.class);
		System.out.println(s);
		
		//scope≤‚ ‘
		student s1 = ac.getBean("s",student.class);
		System.out.println(s==s1);
	}
}
