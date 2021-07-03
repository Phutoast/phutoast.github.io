---
layout: page
permalink: /others/
---

<section class="post-list">
  <div class="container">
    {% assign topics =  site.others | map: 'subject' | join: ','  | split: ',' | uniq %}
    {% for topic in topics %}
      <h4>{{ topic }}</h4>
      {% for post in site.others %}
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


