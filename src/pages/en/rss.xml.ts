import { getRssString } from '@astrojs/rss';
import { getDescription } from '@/utils/index'
import { type SiteLocale } from '@/i18n/config';
import { getDictionary } from '@/i18n/dictionaries';
import { localizePath } from '@/i18n/routes';
import { getBlogPosts } from '@/utils/blogCollection';

const locale: SiteLocale = "en";
const dict = getDictionary(locale);

export async function GET(context: any) {
	const posts = await getBlogPosts(locale);
	const res = await getRssString({
		title: dict.site.title,
		description: dict.site.description,
		site: context.site,
		items: posts.filter(i => !i.data.hide).map((post) => ({
			title: post.data.title,
			pubDate: post.data.updated || post.data.date,
			description: getDescription(post),
			link: localizePath(`/article/${post.data.id}`, locale)
		})).sort((a: any, b: any) => (new Date(b.pubDate).getTime() - new Date(a.pubDate).getTime())),
	});
	const xmlHead = '<?xml version="1.0" encoding="UTF-8"?>';
	const xmlMain = res.replace(xmlHead, `${xmlHead}<?xml-stylesheet type="text/xsl" href="/rss.xsl" ?>`).replace(/\/<\/link>/g, '</link>');
	return new Response(xmlMain, { headers: { 'Content-Type': 'application/xml' } });
}
