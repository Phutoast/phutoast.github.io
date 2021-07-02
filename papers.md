---
layout: page
title: 
permalink: /papers/
---

<section class="post-list">
  <div class="container">
    {% for post in site.notes %}
      <article class="post-item">
        <span class="post-meta date-label">{{ post.conf }}</span>
        <div class="article-title"><a class="post-link" href="{{ post.url | prepend: site.baseurl | prepend: site.url }}">{{ post.title }}</a></div>
      </article>
    {% endfor %}
  </div>

</section>


