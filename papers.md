---
layout: page
permalink: /papers/
---

### Paper Notes

<section class="post-list">
  <div class="container">
    {% assign topics =  site.notes | map: 'topic' | join: ','  | split: ',' | uniq %}
    {% for topic in topics %}
      <h4>{{ topic }}</h4>
      {% for post in site.notes %}
        {% if post.topic == topic %}
        <article class="post-item">
          <span class="post-meta date-label">{{ post.conf }}</span>
          <div class="article-title"><a class="post-link" href="{{ post.url | prepend: site.baseurl | prepend: site.url }}">{{ post.title }}</a></div>
        </article>
        {% endif %}
      {% endfor %}
      <hr>
    {% endfor %}
  </div>

</section>


