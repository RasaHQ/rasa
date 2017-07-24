# Facebook Page Post Scraper

![](/examples/fb_scraper_data.png)

A tool for gathering *all* the posts and comments of a Facebook Page (or Open Facebook Group) and related metadata, including post message, post links, and counts of each reaction on the post. All this data is exported as a CSV, able to be imported into any data analysis program like Excel.

The purpose of the script is to gather Facebook data for semantic analysis, which is greatly helped by the presence of high-quality Reaction data. Here's quick examples of a potential Facebook Reaction data visualization using data from [CNN's Facebook page](https://www.facebook.com/cnn/):

![](/examples/reaction-example-2.png)

## Usage

### Scrape Posts From Public Page

The Page data scraper is implemented as a Python 2/3 script in `get_fb_posts_fb_page.py`; fill in the App ID and App Secret of a Facebook app you control (I strongly recommend creating an app just for this purpose) and the Page ID of the Facebook Page you want to scrape at the beginning of the file. Then run the script by `cd` into the directory containing the script, then running `python get_fb_posts_fb_page.py` or `python3 get_fb_posts_fb_page.py`.

### Scrape Posts from Open Group

To get data from an Open Group, use the `get_fb_posts_fb_group.py` script with the App ID and App Secret filled in the same way. However, the `group_id` is a *numeric ID*. For groups without a custom username, the ID will be in the address bar; for groups with custom usernames, to get the ID, do a View Source on the Group Page, search for the phrase `"entity_id"`, and use the number to the right of that field. For example, the `group_id` of [Hackathon Hackers](https://www.facebook.com/groups/hackathonhackers/) is 759985267390294.

![](/examples/entity.png)

### Scrape Comments From Page/Group Posts

To scrape all the user comments from the posts, create a CSV using either of the above scripts, then run the `get_fb_comments_from_fb.py` script, specifying the Page/Group as the `file_id`. The output includes the original `status_id` where the comment is located so you can map the comment to the original Post with a `JOIN` or `VLOOKUP`, and also a `parent_id` if the comment is a reply to another comment.

Keep in mind that large pages such as CNN have *millions* of comments, so be careful! (scraping throughput is approximately 87k comments/hour)

## Privacy

This scraper can only scrape public Facebook data which is available to anyone, even those who are not logged into Facebook. No personally-identifiable data is collected in the Page variant; the Group variant does collect the name of the author of the post, but that data is also public to non-logged-in users. Additionally, the script only uses officially-documented Facebook API endpoints without circumventing any rate-limits.

Note that this script, and any variant of this script, *cannot* be used to scrape data from user profiles. (and the Facebook API specifically disallows this use case!)

## Known Issues

* UTF-16 text (CJK) sometimes fails.
* GIFs in comments will not appear for an App access_token. (it requires a User access_token for no apparent reason).

## Maintainer

Max Woolf ([@minimaxir](http://minimaxir.com))

*Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

For more information on how the script was originally created, and some tips on how to create similar scrapers yourself, see my blog post [How to Scrape Data From Facebook Page Posts for Statistical Analysis](http://minimaxir.com/2015/07/facebook-scraper/).

## Credits

[Peeter Tintis](https://github.com/Digitaalhumanitaaria), whose [fork](https://github.com/Digitaalhumanitaaria/facebook-page-post-scraper/blob/master/get_fb_posts_fb_page.py) of this repo implements code for finding separate reaction counts per [this Stack Overflow answer](http://stackoverflow.com/a/37239851).

[Marco Goldin](https://github.com/marcogoldin) for the Python 3.5 fork.

## License

MIT

If you do find this script useful, a link back to this repository would be appreciated. Thanks!
