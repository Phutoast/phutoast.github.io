---
layout: page
permalink: /studies/
---

### Study Notes

<section class="post-list">
  <div class="container">
    {% assign topics =  site.studies | map: 'subject' | join: ','  | split: ',' | uniq %}
    {% for topic in topics %}
      <h4>{{ topic }}</h4>
      {% for post in site.studies %}
        {% if post.subject == topic %}
        <article class="post-item">
          <div class="article-title"><a class="post-link" href="{{ post.url | prepend: site.baseurl | prepend: site.url }}">{{ post.title }}</a></div>
        </article>
        {% endif %}
      {% endfor %}
      <hr>
    {% endfor %}
  </div>
</section>


