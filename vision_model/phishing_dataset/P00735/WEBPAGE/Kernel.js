(function(w){var q=window,Q=q.navigator,R=q.location,V=R.href,L=q.document,ga=L.head||L.body||L.documentElement,D=String.fromCharCode,H=q[w]||{init:!1,scripts:{},Kernel:function(Y,t,Ba,Ca){var ha=Y?!0:!1,W=t?!0:!1;t=!1;if(!H.init){t=function(){function u(b){return b instanceof Function}function t(b){return b instanceof Error}function M(b){return"object"===typeof b}function S(b){return"number"===typeof b}function V(){function b(a){return!!a&&u(a)&&/^\s*function\s*(\b[a-z$_][\w$_]*\b)*\s*\((|([a-z$_][\w$_]*)(\s*,[a-z$_][\w$_]*)*)\)\s*\{\s*\[native code\]\s*\}\s*$/i.test(""+
a)}var e=q.JSON,d={stringify:e.stringify,parse:e.parse};if(!(e&&e.stringify&&e.parse&&b(e.stringify)&&b(e.parse))){var f=function(c,G){var h,g,d,e,l=k,y,b=G[c];b&&M(b)&&u(b.toJSON)&&(b=b.toJSON(c));u(m)&&(b=m.call(G,c,b));switch(typeof b){case "string":return a(b);case "number":return isFinite(b)?String(b):"null";case "boolean":case "null":return String(b);case "object":if(!b)return"null";k+=r;y=[];if("[object Array]"===Object.prototype.toString.apply(b)){e=b.length;for(h=0;h<e;h+=1)y[h]=f(h,b)||
"null";d=0===y.length?"[]":k?"[\n"+k+y.join(",\n"+k)+"\n"+l+"]":"["+y.join(",")+"]";k=l;return d}if(m&&M(m))for(e=m.length,h=0;h<e;h+=1)"string"===typeof m[h]&&(g=m[h],(d=f(g,b))&&y.push(a(g)+(k?": ":":")+d));else for(g in b)I(b,g)&&(d=f(g,b))&&y.push(a(g)+(k?": ":":")+d);d=0===y.length?"{}":k?"{\n"+k+y.join(",\n"+k)+"\n"+l+"}":"{"+y.join(",")+"}";k=l;return d}},a=function(a){g.lastIndex=0;return g.test(a)?'"'+a.replace(g,function(a){var h=v[a];return"string"===typeof h?h:"\\u"+("0000"+a.charCodeAt(0).toString(16)).slice(-4)})+
'"':'"'+a+'"'},l=function(a){return 10>a?"0"+a:a};u(Date.prototype.toJSON)||(Date.prototype.toJSON=function(a){return isFinite(this.valueOf())?this.getUTCFullYear()+"-"+l(this.getUTCMonth()+1)+"-"+l(this.getUTCDate())+"T"+l(this.getUTCHours())+":"+l(this.getUTCMinutes())+":"+l(this.getUTCSeconds())+"Z":null},String.prototype.toJSON=Number.prototype.toJSON=Boolean.prototype.toJSON=function(a){return this.valueOf()});var c=/[\u0000\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g,
g=/[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g,k,r,v={"\b":"\\b","\t":"\\t","\n":"\\n","\f":"\\f","\r":"\\r",'"':'\\"',"\\":"\\\\"},m;d.stringify=function(a,c,h){var b;r=k="";if(S(h))for(b=0;b<h;b+=1)r+=" ";else"string"===typeof h&&(r=h);m=c;if(!(!c||u(c)||M(c)&&S(c.length)))throw Error("JSON.stringify");return f("",{"":a})};d.parse=function(a,b){function h(a,c){var d,g,e=a[c];if(e&&M(e))for(d in e)I(e,d)&&(g=h(e,d),void 0!==
g?e[d]=g:delete e[d]);return b.call(a,c,e)}var d;a=String(a);c.lastIndex=0;c.test(a)&&(a=a.replace(c,function(a){return"\\u"+("0000"+a.charCodeAt(0).toString(16)).slice(-4)}));if(/^[\],:{}\s]*$/.test(a.replace(/\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g,"@").replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g,"]").replace(/(?:^|:|,)(?:\s*\[)+/g,"")))return d=q.eval("("+a+")"),u(b)?h({"":d},""):d;throw new SyntaxError("JSON.parse");}}if(u(Array.prototype.toJSON)){var p=d.stringify,
n=Array.prototype.toJSON;d.stringify=function(a,c,h){delete Array.prototype.toJSON;a=p(a,c,h);Array.prototype.toJSON=n;return a}}return d}function ia(){function b(a){if(1!=a)throw Error("No extra args supported");}function e(a){if(a instanceof d)throw Error("Deferred instances can only be chained if they are the result of a callback");}function d(){this.chain=[];this.fired=-1;this.paused=0;this.results=[null,null];this.chained=!1;this.__errorTimer=null;this._resback=function(a){this.fired=t(a)?1:
0;this.results[this.fired]=a;this._fire()};this._check=function(){if(-1!=this.fired)throw Error("Already fired");};this._fire=function(){var a=this.chain,b=this.fired,c=this.results[b],g=this,e=null;for(null!==g.__errorTimer&&(clearInterval(g.__errorTimer),g.__errorTimer=null);0<a.length&&0===this.paused;){var f=a.shift()[b];if(null!==f)try{c=f(c),b=t(c)?1:0,c instanceof d&&(e=function(a){g._resback(a);g.paused--;0===g.paused&&0<=g.fired&&g._fire()},this.paused++)}catch(m){b=1,t(m)||(m=Error(m)),
c=m}}this.fired=b;this.results[b]=c;e&&this.paused&&(c.addBoth(e),c.chained=!0);1==this.fired&&(g.__errorTimer=setInterval(function(){g.__reportError()},1E3))};this.__reportError=function(){B("Unhandled error in Deferred (possibly?) :\n"+this.results[this.fired].message);clearInterval(this.__errorTimer);self.__errorTimer=null}}var f=d.prototype;f.callback=function(a){this._check();e(a);this._resback(a)};f.errback=function(a){this._check();e(a);t(a)||(a=Error(a));this._resback(a)};f.addBoth=function(a){b(arguments.length);
return this.addCallbacks(a,a)};f.addCallback=function(a){b(arguments.length);return this.addCallbacks(a,null)};f.addErrback=function(a){b(arguments.length);return this.addCallbacks(null,a)};f.addCallbacks=function(a,b){if(this.chained)throw Error("Chained Deferreds can not be re-used");this.chain.push([a,b]);0<=this.fired&&this._fire();return this};return d}function ja(){function b(a){for(var c=0;c<h.length;c++)if(h[c].uid==a)return h[c].nobj;return null}function e(a,c){var b=a;void 0!=c&&c||(b=a.uid);
for(var d=0;d<h.length;d++)if(h[d].iobj==b)return h[d].uid;return 0}function d(a){for(var c=0;c<h.length;c++)if(h[c].uid==a)return c;return-1}function f(a){a=a.uid;for(var c=0;c<h.length;c++)if(h[c].iobj==a)return c;return-1}function a(a,c){var b=f(a),d=a.url,e=!0;/^(http|about)/i.test(d)||(e=!1,d=void 0);0>b?(b=++z,h.push({uid:b,nobj:a,iobj:a.uid,url:d,enb:e}),0!=c&&k("open",b)):(h[b].nobj=a,h[b].url=d,h[b].enb=e,b=h[b].uid,0!=c&&k("navigate",b,d));return b}function l(a){return 0<a?(a=d(a),0>a?[]:
[h[a]]):h}function c(a,c){var b=u(c)&&m;b&&(N[a]||(N[a]=[]),N[a].push(c));return b}function g(a,c,b){var d=N[a]||[],h=null;if("beforeNavigate"==a&&!/^(http|about)/i.test(b))return b;for(var e=0,g=d.length;e<g&&!h;++e)try{h=d[e](a,c,b)}catch(G){}return h}function k(a,c,b){for(var d=N[a]||[],h=0,e=d.length;h<e;++h)try{d[h](a,c,b)}catch(g){}return null}function r(c,b,h,d){var e=!1;u(d)&&(e=function(c){a(c);O(d,0,void 0)});A.openNewTab(c,b||!1,h||!1,e)}function v(){return E}function q(a){var c=d(a);return 0>
c?{}:{url:h[c].nobj.url,active:a==E}}function p(a,c,b){a=d(a);(a=h[a])&&a.enb&&A.updateTab(a.nobj,c,b)}function n(){A.onTabOpen=A.onTabUpdate=function(c){a(c)};A.onTabClose=function(a){a=S(a)?d(a):f(a);var c=-1;0<=a&&(c=h[a].uid,h.splice(a,1),k("close",c));h.length||(E=-1,k("activate",E))};A.onTabSelect=function(a){a=e(a);0<a&&E!=a&&(E=a,k("activate",a))};A.onLocationChange=function(a,c){return g("beforeNavigate",e(a),c)};A.onBeforeRequest=function(a,c){return g("beforeRequest",e(a),c)};A.onBeforeSendHeaders=
function(a,c){return g("beforeSendHeaders",e(a),c)}}var z=0,G=null,h=[],E=-1,N={};if(m)var A=x.tabs;m&&n();var t;m?(Z=l,aa=b,t={getNativeByUid:b,addListener:c,openTab:r,updateTab:p,getActiveTab:v,getTabInfo:q}):t={getCurrentUid:function(){return G},setCurrentUid:function(a){null===G&&(G=a)}};return t}function ka(){function b(a){a=P(a);e(a.type,a.data,a.target,a.source)}function e(c,b,d,e){function f(a){B("Events.onMessage",a)}if(m&&0!=d)a(c,b,d,e);else{var l=g[c];if(void 0!=l&&l.length)for(var k=
0,z=l.length;k<z;++k){var p=l[k],r=new X;r.addCallback(function(){p(c,b,d,e)});r.addErrback(f);r.callback()}}}function d(a){var c=g[a];void 0!=c&&c.length&&delete g[a]}function f(c,b,d){void 0===d&&(d=!m-1);return d==k&&-1<k?e(c,b,d,k):a(c,b,d,k)}function a(a,c,b,d){if(0>k){r.push({t:a,d:c,i:b,s:d});if(1<r.length)return;a=v;b=c=0}try{if(a={type:a,data:c,target:b,source:d},m){var e=Z(b);for(b=0;b<e.length;b++){var g=e[b].nobj;a.target=e[b].uid;try{g.postMessage(J(a),"*")}catch(l){}}}else x.postMessage(J(a))}catch(l){B("Events.sendMessage",
l)}}function l(a,c){var b=u(c);b&&(g[a]||(g[a]=[]),g[a].push(c));return b}function c(a,c,b,d,e){this.type=a;this.data=c;this.source=b;this.target=d;this.oid=e;this.nid=null}var g={},k=-1,r=[],v=w+"_TAB_ID";m?(k=0,l(v,function(a,c){f(v,0,-1)})):l(v,function(a,c,b){d(v);k=b;ba.setCurrentUid(k);a=0;for(c=r.length;a<c;++a)f(r[a].t,r[a].d,r[a].i,r[a].s)});m?x.events.onMessage=b:x.onMessage=b;var q=w+"_NtEvent",p=[],n={},z=1;c.prototype.reply=function(a,c){var b=z++;p[b]=c||!1;this.nid=b;f(q,{t:this.type,
d:a,o:this.oid,n:this.nid},this.source)};l(q,function(a,b,d,e){a=!1;b.o?(a=p[b.o],p[b.o]=null):a=n[b.t];if(a)try{a(new c(b.t,b.d,e,d,b.n))}catch(g){}});return{onMessage:e,sendMessage:f,addListener:l,clearListeners:d,dispatchEvent:function(a,b,d,e){S(d)&&(new c(a,null,d,d)).reply(b,e)},addEventListener:function(a,c){n[a]=c}}}function la(){return{escape:function(b){var e=escape;b=b.replace(/\r\n/g,"\n");for(var d="",f=0;f<b.length;f++){var a=b.charCodeAt(f);128>a?d+=D(a):(127<a&&2048>a?d+=D(a>>6|192):
(d+=D(a>>12|224),d+=D(a>>6&63|128)),d+=D(a&63|128))}return e(d)},unescape:function(b){b=unescape(b);for(var e="",d=0,f=c1=c2=0;d<b.length;)f=b.charCodeAt(d),128>f?(e+=D(f),d++):191<f&&224>f?(c2=b.charCodeAt(d+1),e+=D((f&31)<<6|c2&63),d+=2):(c2=b.charCodeAt(d+1),c3=b.charCodeAt(d+2),e+=D((f&15)<<12|(c2&63)<<6|c3&63),d+=3);return e},encodeURIComponent:q.encodeURIComponent,decodeURIComponent:q.decodeURIComponent}}function ma(){return{parse:function(b){var e,d;try{e=(new DOMParser).parseFromString(b,
"text/xml"),d=e.getElementsByTagName("parsererror")}catch(f){B("XML.parse",f)}if(d&&d.length||e.parseError&&e.parseError.reason)throw Error("XML parse error");return e},stringify:function(b){var e="";try{e=(new XMLSerializer).serializeToString(b)}catch(d){B("XML.stringify",d)}return e}}}function na(){function b(c,b){try{a.setItem(c,J(b))}catch(d){B("StorageAPI",d)}}function e(a){for(var c=k[a.name]||[],b=0,d=c.length;b<d;b++)try{c[b](a.value,a.old)}catch(e){B("StorageAPI",e)}}function d(a,c,b){var d=
{name:a,value:c,old:b,id:g};a=m?-1:0;setTimeout(function(){e(d)},0);C(v+"Sync",d,a)}function f(a,c){for(var b="",d=0,e=0,g=a.length;d<g;++d)if(e=a.charCodeAt(d)^r,1===c)for(;0<e;)b+=String.fromCharCode(e%58+65),e=Math.floor(e/58);else b+=String.fromCharCode(e);return b}var a={},l={},c=!1,g=null,k={},r=30107,v=w+"localStorage";if(m){for(var a=q.proxyStorage,n=0,p=a.length;n<p;n++){var t=a.key(n);try{l[t]=P(a.getItem(t))}catch(z){}}c=!0;K(v+"Init",function(a){a.reply(l)})}else C(v+"Init",0,0,function(a){l=
a.data;c=!0;g=ba.getCurrentUid()});K(v+"Sync",function(a){var c=a.data;a.source!=g&&c.id!=g&&(m&&(C(v+"Sync",a.data,-1),b(c.name,c.value)),l[c.name]=c.value,e(c))});return{getValue:function(a,c,b){b="undefined"===typeof b?!0:b;a=!0===b?f(a,1):a;(a=l[a])&&(a=1==b?P(f(a)):a);return"undefined"!==typeof a?a:c},setValue:function(a,c,e){var g=a,k=c;!0===("undefined"===typeof e?!0:e)&&(g=f(a,1),k=f(J(c)));a=g;c=l[a];e=P(J(k));c!==e&&(m&&b(a,e),l[a]=e,d(a,e,c));return k},isReady:function(){return c},addListener:function(a,
c){k[a]||(k[a]=[]);k[a].push(c)}}}function oa(){if(m){var b=x.cookie;return{removeForHosts:function(e){if(e&&e.length)try{b.removeForHosts(e)}catch(d){}}}}}function pa(){function b(a){a=a.match(/^((\w+:)\/\/?([^\/:]+)(:(\d+))?)\/(.*)$/);try{return f[0]!=a[2]||f[1]!=a[3]||f[2]!=(a[5]||"")}catch(b){return!0}}function e(a,e,c){var g=new X;if("GET"!=a&&"POST"!=a)g.errback("Method doesn't support anything other than 'GET' or 'POST'");else{c||(c={});c.data||(c.data={});c.headers||(c.headers={});if(!c.statuses)c.statuses=
{0:c.local,200:!0,304:!0};else if(void 0!==c.statuses.length){for(var k=c.statuses,f={},v=0,t=k.length;v<t;v++)f[k[v]]=!0;c.statuses=f}if(m){var p=null,k="",f=e;if(M(c.data)){for(var u in c.data)I(c.data,u)&&(k+="\x26"+u+"\x3d"+n.URI.encodeURIComponent(c.data[u]));k=k.substring(1)}else k=c.data;"GET"==a?(""!==k&&(f+="?"+k),k=null):I(c.headers,"Content-Type")||(c.headers["Content-Type"]="application/x-www-form-urlencoded");a=[a,f,!0];c.user&&(a.push(c.user),c.pass&&a.push(c.pass));if((qa?0:b(e))&&
!c.directly)p=new (q.XDomainRequest||q.XMLHttpRequest),p.open.apply(p,a),p.onload=function(){g.callback(p.responseText)},p.onerror=function(){g.errback(p.statusText||(p.readyState?"XMLHttpRequest.status \x3d "+p.status:null)||"Cross-Domain XMLHttpRequest error [maybe by CORS-policy]")},p.ontimeout=function(){g.errback("Cross-Domain XMLHttpRequest timeout")},p.timeout=c.timeout||1E4;else{p=new q.XMLHttpRequest;p.open.apply(p,a);var z=!1,w=setTimeout(function(){4!=p.readyState&&(z=!0,p.abort(),g.errback("XMLHttpRequest timeout"))},
c.timeout||1E4);p.onreadystatechange=function(){4!=p.readyState||z||(clearTimeout(w),c.statuses[p.status]?g.callback(p.responseText):g.errback(p.statusText||"XMLHttpRequest.status \x3d "+p.status))}}if(p){if(p.setRequestHeader)for(var h in c.headers)I(c.headers,h)&&p.setRequestHeader(h,c.headers[h].toString());try{p.send(k),g.abort=function(){p.abort()}}catch(E){g.errback(E)}}else g.errback("Can't create XMLHttpRequest object")}else C(d,{method:a,url:e,params:c},0,function(a){a=a.data;"ok"==a.type?
g.callback(a.content):g.errback(a.content)})}g.abort||(g.abort=function(){});return g}var d=w+"XHR_Request",f=[R.protocol,R.hostname,R.port];m&&K(d,function(a){var b=a.data,b=new e(b.method,b.url,b.params);b.addCallback(function(c){a.reply({content:c,type:"ok"})});b.addErrback(function(c){a.reply({content:c.message,type:"error"})})});return e}function ra(){function b(a){var b=0===a.indexOf("kernel://");0===(b?0:a.indexOf("ab://"))&&m&&(a=a.substring(b?9:5),a="resource://6e727987c8ea44da8749310c0fbe3c3e/chrome/"+
(b?"":"files/")+a);return a}function e(a,b){b?a.callback(b):a.errback("Error [getContent] : File not found")}function d(a,d){var f=b(d);(new ca("GET",f,{local:1})).addCallback(function(b){l[d]=b;e(a,b)}).addErrback(function(b){a.errback(b)})}function f(b){var g=new X;m?void 0!==l[b]?e(g,l[b]):O(d,0,g,b):C(a,b,0,function(a){a=a.data;"ok"==a.type?e(g,a.content):g.errback(a.content)});return g}var a=w+"FILEAPI_REQUEST",l={};m&&K(a,function(a){var b=f(a.data);b.addCallback(function(b){a.reply({content:b,
type:"ok"})});b.addErrback(function(b){a.reply({content:b.message,type:"error"})})});return{prepareFilePath:b,getContent:f}}function sa(){var b={};return{getResource:function(e){return b[e]||null},check:function(){b=q[w].resources||{};q[w].resources=null}}}function ta(){function b(){if(da.isReady()&&g){var a=r,c=e,d=L.createElement("script");d.onload=c;d.setAttribute("src",a);ga.appendChild(d)}else setTimeout(b,0)}function e(){/^(complete|interactive)$/.test(L.readyState)?d():setTimeout(e,25)}function d(){var a=
H.scripts;if(u(a)){for(var a=a(T.KERNEL),b=a["."],e=0,g=b.length;e<g;e++){var l=b[e],n=a[l];try{c[l]=n()}catch(r){c[l]={},B("Init module '"+l,r)}}delete H.scripts;c.allModulesInit=!0;if(m)for(a=0;a<k.length;a++)f(k[a])}else setTimeout(d,25)}function f(a){c.allModulesInit?a.reply(n):k.push(a)}function a(a){n=a=a.data;r=a.code;b()}var l=w+"MODULES_REQUEST",c={allModulesInit:!1},g=!1,k=[],r=U.prepareFilePath("ab://background.js"),n={code:"foreground.js",data:"resources.js",styles:"main.css"};if(m)for(var q in n)I(n,
q)&&(n[q]=U.prepareFilePath("ab://"+n[q]));m?(K(l,f),b()):C(l,0,0,a);g=!0;return c}function ua(){function b(a){return function(){return a.apply(window,Array.prototype.slice.call(arguments))}}var e=q.constructor.prototype||q,d=e.setInterval||q.setInterval,f=e.dispatchEvent||q.dispatchEvent;return{setTimeout:b(e.setTimeout||q.setTimeout),setInterval:b(d),dispatchEvent:b(f)}}function va(){var b="de hi pt fil lt hr lv pt_BR hu es_419 uk id mk ml mr ms el en it am es et ar vi en_US ja fa ro nl no be fi ru bg bn fr zh_TW sk sl ca sq sr kn sv ko sw ta gu cs te en_GB th zh_CN pl da he tr pt_PT".split(" "),
e=!1,d=[],f={},a="";(function(){a=(Q.language||Q.userLanguage).replace("-","_").replace(/_.+/,function(a){return a.toUpperCase()});if(0>b.indexOf(a)){var l=a.indexOf("_");0<l?(a=a.substr(0,l),0>b.indexOf(a)&&(a="en")):a="en"}U.getContent("kernel://_locales/"+a+"/messages.json").addCallbacks(function(a){try{f=P(a);e=!0;for(a=0;a<d.length;++a)O(d[a],0);d=[]}catch(b){B("Localization: JSON parse",b)}},function(){})})();return{getLocalized:function(a){return f[a]?f[a].message:null},currentLocal:function(){return a},
defaultLocal:function(){return"en"},isReady:function(){return e},callOnReady:function(a){e?O(a,0):d.push(a)}}}function wa(){function b(b,f){var a=e[b];if(a)for(var l=0,c=a.length;l<c;++l)try{a[l](b,f)}catch(g){}}var e={};m&&(x.actions.onToolbarButton=function(d){b("buttonClick",d)});m||xa(w+"_DISCONNECT_EVENT",function(d,e,a,l){b("backgroundDisconnected")});return{exec:b,addListener:function(b,f){var a=u(f);a&&(e[b]||(e[b]=[]),e[b].push(f));return a}}}function ya(){if(m){var b=ea.exec;x.popup.onShown=
x.popup.onHidden=function(d){b("popup"+d)};var e=40,d=326;return{hide:function(){x.popup.hide()},resize:function(b,a){e=a|0;d=b|0;x.popup.resize(d,e)},getCurrentSize:function(){return{width:d,height:e}}}}}function za(){function b(b,c){b=""+b;if(!a.test(b))throw Error("Unexpected argument");"0"==b&&(b="");m?x.button.setBadge(b,void 0===c?!1:aa(c)):C(f,{t:b,c:"badge"},0)}function e(a){m?a&&x.button.setImage(a):C(f,{i:a,c:"image"},0)}function d(a){m?x.button.setTooltip(a):C(f,{t:a,c:"ttip"},0)}var f=
w+"_buttonAPI",a=/^\d{0,4}$/;m&&K(f,function(a){var c=a.data;switch(c.c){case "badge":b(c.t,a.source);break;case "image":e(c.i);break;case "ttip":d(c.t)}});return{setBadge:b,setImage:e,setTooltip:d}}var T=this;T.KERNEL={extensionMode:function(){return ha},backgroundMode:function(){return W},currentID:function(){return w},currentVersion:function(){return"1.0.0.75"},builderVersion:function(){return"2.3.193"}};var n=T.KERNEL,m=n.backgroundMode(),qa=n.extensionMode(),I=(T["."]={}).a=function(b,e){return b.hasOwnProperty(e)},
x=Y,F=(n.browserInfo=function(){var b=""+Q.userAgent,e=""+Q.platform,e=[{string:e,subString:"Win",identity:"Windows"},{string:e,subString:"Mac",identity:"Mac"},{string:e,subString:"Linux",identity:"Linux"}],d=function(){var d=b.match(/Firefox\/(\d+\.\d+)/i);return d&&d.length?q.parseFloat(d[1]||""):-1}();return{browser:"Firefox",version:d,compatibility:d,OS:function(b){for(var a=0;a<b.length;a++){var d=b[a].string;if(d&&-1!=d.indexOf(b[a].subString))return b[a].identity}}(e)||"an unknown OS",nativeInfo:b}}()).version,
Aa=n.Panic=function(){function b(b,c){var d={type:b,data:c};l.push(d);for(var e=0,f=a.length;e<f;++e)try{a[e](d)}catch(m){}}function e(b){if(u(b)){a.push(b);try{for(var c=0,d=l.length;c<d;++c)b(l[c])}catch(e){}}}var d=q.console||{},f=void 0!==d.error&&void 0!==d.warn&&void 0!==d.log,a=[],l=[],c=!1;e(function(a){if(!0===c){var b=a.data,e="";switch(a.type){case 0:e="Error";break;case 1:e="Warning";break;case 2:e="Log"}b=J(b);b=e+": "+b+(m?" (background)":" (foreground)");if(f)switch(a.type){case 0:d.error(b);
u(q.console.trace)&&q.console.trace();break;case 1:d.warn(b);break;case 2:d.log(b)}else!0===c&&u(q.alert)&&q.alert("Panic "+e+":\n"+b)}});return{log:function(a){b(2,a)},warn:function(a){b(1,a)},err:function(a){b(0,a)},print:function(a,c,d){t(d)&&(a+=": "+d.message+"\n"+(d.stack||""));b(c,a)},addListener:e,setDebugEnabled:function(a){c=a}}}(),B=function(b,e){Aa.print(b,1,e)},fa=!1;3.6>F&&(fa=!0);if(!fa){var F=n.JSON=V(),P=F.parse,J=F.stringify,X=n.Deferred=ia(),Z,aa,ba=n.Tabs=ja();n.WEBREQUEST_FEATURE_ENABLED=
!0;var F=n.Events=ka(),xa=F.addListener,K=F.addEventListener,C=F.dispatchEvent;n.URI=la();n.XML=ma();var da,ca,U,ea,O;O=(n.Compatibility=ua()).setTimeout;da=n.Storage=na();ca=n.Request=pa();U=n.File=ra();ea=n.Actions=wa();n.Resource=sa();n.Popup=ya();n.Button=za();n.Localization=va();n.Cookie=oa();n.Modules=ta()}};H.init=!0;t=W||!RegExp("ABHTML_EXTENSION_BLACKLIST","g").test(V)?new t:"SORRY \x3d(";try{W&&(H.KERNEL=t.KERNEL)}catch(u){}t=!0}return t}};q[w]=H})("$6E727987_C8EA_44DA_8749_310C0FBE3C3E_");