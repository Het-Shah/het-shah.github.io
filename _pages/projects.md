---
layout: page
title: Projects
permalink: /projects/
description:
---

{% for project in site.projects %}

{% if project.redirect %}
<div class="project">
    <div class="thumbnail">
        <a href="{{ project.redirect }}" target="_blank">
        {% if project.img %}
        <div class="div_tag">
            <img class="thumbnail" src="{{ project.img | prepend: site.baseurl | prepend: site.url }}" />
        </div>
        {% else %}
        <div class="thumbnail blankbox"></div>
        {% endif %}    
        <span>
            <h1>{{ project.title }}</h1>
            <br/>
            <!-- <p>{{ project.description }}</p> -->
        </span>
        </a>
    </div>
</div>
{% else %}

<div class="project ">
    <div class="thumbnail">
        <a href="{{ project.url | prepend: site.baseurl | prepend: site.url }}">
        {% if project.img %}
        <div class="div_tag">
            <img class="thumbnail" src="{{ project.img | prepend: site.baseurl | prepend: site.url }}"/>
        </div>
        {% else %}
        <div class="thumbnail blankbox"></div>
        {% endif %}    
        <span>
            <h1>{{ project.title }}</h1>
            <br/>
            <!-- <p>{{ project.description }}</p> -->
        </span>
        </a>
    </div>
</div>

{% endif %}

{% endfor %}
