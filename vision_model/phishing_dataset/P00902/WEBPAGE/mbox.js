var mboxCopyright="Copyright 1996-2014 Adobe Systems Incorporated. All rights reserved.";var TNT=TNT||{};TNT.a=TNT.a||{};TNT.a.nestedMboxes=[];TNT.a.isDomLoaded=false;TNT.getGlobalMboxName=function(){return"Global_Mbox"};TNT.getGlobalMboxLocation=function(){return""};TNT.isAutoCreateGlobalMbox=function(){return false};TNT.getClientMboxExtraParameters=function(){return""};TNT.a.b=function(){var e={}.toString;var f=window.targetPageParams;var g=null;if(typeof(f)==="undefined"||e.call(f)!=="[object Function]"){return[]}try{g=f()}catch(h){}if(g===null){return[]}if(e.call(g)==="[object Array]"){return g}if(e.call(g)==="[object String]"&&g.length>0){var c=g.split("&");for(var d=0;d<c.length;d++){if(c[d].indexOf("=")<=0){c.splice(d,1);continue}c[d]=decodeURIComponent(c[d])}return c}if(e.call(g)==="[object Object]"){return TNT.a.i([],g)}return[]};TNT.a.i=function(m,n){var h=[];var i=m.join(".");var j=undefined;for(o in n){if(!n.hasOwnProperty(o)){continue}j=n[o];if(typeof j==="object"){m.push(o);var k=TNT.a.i(m,j);m.pop();for(var l=0;l<k.length;l++){h.push(k[l])}continue}h.push((i.length>0?i+".":"")+o+"="+j)}return h};mboxUrlBuilder=function(c,d){this.q=c;this.r=d;this.s=new Array();this.t=function(a){return a};this.v=null};mboxUrlBuilder.prototype.addNewParameter=function(c,d){this.s.push({name:c,value:d});return this};mboxUrlBuilder.prototype.addParameterIfAbsent=function(e,f){if(f){for(var g=0;g<this.s.length;g++){var h=this.s[g];if(h.name===e){return this}}this.checkInvalidCharacters(e);return this.addNewParameter(e,f)}};mboxUrlBuilder.prototype.addParameter=function(e,f){this.checkInvalidCharacters(e);for(var g=0;g<this.s.length;g++){var h=this.s[g];if(h.name===e){h.value=f;return this}}return this.addNewParameter(e,f)};mboxUrlBuilder.prototype.addParameters=function(d){if(!d){return this}for(var f=0;f<d.length;f++){var e=d[f].indexOf("=");if(e==-1||e==0){continue}this.addParameter(d[f].substring(0,e),d[f].substring(e+1,d[f].length))}return this};mboxUrlBuilder.prototype.setServerType=function(b){this.C=b};mboxUrlBuilder.prototype.setBasePath=function(b){this.v=b};mboxUrlBuilder.prototype.setUrlProcessAction=function(b){this.t=b};mboxUrlBuilder.prototype.buildUrl=function(){var k=this.v?this.v:"/m2/"+this.r+"/mbox/"+this.C;var l=document.location.protocol=="file:"?"http:":document.location.protocol;var h=l+"//"+this.q+k;var g=h.indexOf("?")!=-1?"&":"?";for(var i=0;i<this.s.length;i++){var j=this.s[i];h+=g+encodeURIComponent(j.name)+"="+encodeURIComponent(j.value);g="&"}return this.H(this.t(h))};mboxUrlBuilder.prototype.getParameters=function(){return this.s};mboxUrlBuilder.prototype.setParameters=function(b){this.s=b};mboxUrlBuilder.prototype.clone=function(){var d=new mboxUrlBuilder(this.q,this.r);d.setServerType(this.C);d.setBasePath(this.v);d.setUrlProcessAction(this.t);for(var c=0;c<this.s.length;c++){d.addParameter(this.s[c].name,this.s[c].value)}return d};mboxUrlBuilder.prototype.H=function(b){return b.replace(/\"/g,"&quot;").replace(/>/g,"&gt;")};mboxUrlBuilder.prototype.checkInvalidCharacters=function(c){var d=new RegExp("('|\")");if(d.exec(c)){throw"Parameter '"+c+"' contains invalid characters"}};mboxStandardFetcher=function(){};mboxStandardFetcher.prototype.getType=function(){return"standard"};mboxStandardFetcher.prototype.fetch=function(b){b.setServerType(this.getType());document.write('<script src="'+b.buildUrl()+'" language="JavaScript"><\/script>')};mboxStandardFetcher.prototype.cancel=function(){};mboxAjaxFetcher=function(){};mboxAjaxFetcher.prototype.getType=function(){return"ajax"};mboxAjaxFetcher.prototype.fetch=function(d){d.setServerType(this.getType());var c=d.buildUrl();this.M=document.createElement("script");this.M.src=c;document.body.appendChild(this.M)};mboxAjaxFetcher.prototype.cancel=function(){};mboxMap=function(){this.N=new Object();this.O=new Array()};mboxMap.prototype.put=function(c,d){if(!this.N[c]){this.O[this.O.length]=c}this.N[c]=d};mboxMap.prototype.get=function(b){return this.N[b]};mboxMap.prototype.remove=function(f){this.N[f]=undefined;var d=[];for(var e=0;e<this.O.length;e++){if(this.O[e]!==f){d.push(this.O[e])}}this.O=d};mboxMap.prototype.each=function(i){for(var h=0;h<this.O.length;h++){var j=this.O[h];var g=this.N[j];if(g){var f=i(j,g);if(f===false){break}}}};mboxMap.prototype.isEmpty=function(){return this.O.length===0};mboxFactory=function(f,j,g){this.T=false;this.R=f;this.S=g;this.U=new mboxList();mboxFactories.put(g,this);this.V=typeof document.createElement("div").replaceChild!="undefined"&&(function(){return true})()&&typeof document.getElementById!="undefined"&&typeof(window.attachEvent||document.addEventListener||window.addEventListener)!="undefined"&&typeof encodeURIComponent!="undefined";this.W=this.V&&mboxGetPageParameter("mboxDisable")==null;var h=g=="default";this.Y=new mboxCookieManager("mbox"+(h?"":("-"+g)),(function(){return mboxCookiePageDomain()})());this.W=this.W&&this.Y.isEnabled()&&(this.Y.getCookie("disable")==null);if(this.isAdmin()){this.enable()}this.Z();this._=mboxGenerateId();this.ab=mboxScreenHeight();this.bb=mboxScreenWidth();this.cb=mboxBrowserWidth();this.db=mboxBrowserHeight();this.eb=mboxScreenColorDepth();this.fb=mboxBrowserTimeOffset();this.gb=new mboxSession(this._,"mboxSession","session",31*60,this.Y);this.hb=new mboxPC("PC",7776000,this.Y);this.L=new mboxUrlBuilder(f,j);this.ib(this.L,h);this.jb=new Date().getTime();this.kb=this.jb;var i=this;this.addOnLoad(function(){i.kb=new Date().getTime()});if(this.V){this.addOnLoad(function(){i.T=true;i.getMboxes().each(function(a){a.nb();a.setFetcher(new mboxAjaxFetcher());a.finalize()});TNT.a.nestedMboxes=[];TNT.a.isDomLoaded=true});if(this.W){this.limitTraffic(100,10368000);this.ob();this.pb=new mboxSignaler(function(a,b){return i.create(a,b)},this.Y)}}};mboxFactory.prototype.isEnabled=function(){return this.W};mboxFactory.prototype.getDisableReason=function(){return this.Y.getCookie("disable")};mboxFactory.prototype.isSupported=function(){return this.V};mboxFactory.prototype.disable=function(d,c){if(typeof d=="undefined"){d=60*60}if(typeof c=="undefined"){c="unspecified"}if(!this.isAdmin()){this.W=false;this.Y.setCookie("disable",c,d)}};mboxFactory.prototype.enable=function(a){if(typeof a==="undefined"||a==null||a!=true||(reason=this.Y.getCookie("disable"),reason=="ccp")){this.W=true;this.Y.deleteCookie("disable")}};mboxFactory.prototype.isAdmin=function(){return document.location.href.indexOf("mboxEnv")!=-1};mboxFactory.prototype.limitTraffic=function(d,c){};mboxFactory.prototype.addOnLoad=function(e){if(this.isDomLoaded()){e()}else{var d=false;var f=function(){if(d){return}d=true;e()};this.xb.push(f);if(this.isDomLoaded()&&!d){f()}}};mboxFactory.prototype.getEllapsedTime=function(){return this.kb-this.jb};mboxFactory.prototype.getEllapsedTimeUntil=function(b){return b-this.jb};mboxFactory.prototype.getMboxes=function(){return this.U};mboxFactory.prototype.get=function(c,d){return this.U.get(c).getById(d||0)};mboxFactory.prototype.update=function(d,e){if(!this.isEnabled()){return}if(!this.isDomLoaded()){var f=this;this.addOnLoad(function(){f.update(d,e)});return}if(this.U.get(d).length()==0){throw"Mbox "+d+" is not defined"}this.U.get(d).each(function(a){a.getUrlBuilder().addParameter("mboxPage",mboxGenerateId());mboxFactoryDefault.setVisitorIdParameters(a.getUrlBuilder(),d);a.load(e)})};mboxFactory.prototype.setVisitorIdParameters=function(g,j){var h="14CF22CE52782FEA0A490D4D@AdobeOrg";if(typeof Visitor=="undefined"||h.length==0){return}var i=Visitor.getInstance(h);if(i.isAllowed()){var f=function(b,d,e){if(i[d]){var a=function(l){if(l){g.addParameter(b,l)}};var c;if(typeof e!="undefined"){c=i[d]("mbox:"+e)}else{c=i[d](a)}a(c)}};f("mboxMCGVID","getMarketingCloudVisitorID");f("mboxMCGLH","getAudienceManagerLocationHint");f("mboxAAMB","getAudienceManagerBlob");f("mboxMCAVID","getAnalyticsVisitorID");f("mboxMCSDID","getSupplementalDataID",j)}};mboxFactory.prototype.create=function(s,m,w){if(!this.isSupported()){return null}var n=this.L.clone();n.addParameter("mboxCount",this.U.length()+1);n.addParameters(m);this.setVisitorIdParameters(n,s);var u=this.U.get(s).length();var q=this.S+"-"+s+"-"+u;var r;if(w){r=new mboxLocatorNode(w)}else{if(this.T){throw"The page has already been loaded, can't write marker"}r=new mboxLocatorDefault(q)}try{var v=this;var x="mboxImported-"+q;var p=new mbox(s,u,n,r,x);if(this.W){p.setFetcher(this.T?new mboxAjaxFetcher():new mboxStandardFetcher())}p.setOnError(function(a,b){p.setMessage(a);p.activate();if(!p.isActivated()){v.disable(60*60,a);window.location.reload(false)}});this.U.add(p)}catch(t){this.disable();throw'Failed creating mbox "'+s+'", the error was: '+t}var y=new Date();n.addParameter("mboxTime",y.getTime()-(y.getTimezoneOffset()*60000));return p};mboxFactory.prototype.getCookieManager=function(){return this.Y};mboxFactory.prototype.getPageId=function(){return this._};mboxFactory.prototype.getPCId=function(){return this.hb};mboxFactory.prototype.getSessionId=function(){return this.gb};mboxFactory.prototype.getSignaler=function(){return this.pb};mboxFactory.prototype.getUrlBuilder=function(){return this.L};mboxFactory.prototype.ib=function(d,c){d.addParameter("mboxHost",document.location.hostname).addParameter("mboxSession",this.gb.getId());if(!c){d.addParameter("mboxFactoryId",this.S)}if(this.hb.getId()!=null){d.addParameter("mboxPC",this.hb.getId())}d.addParameter("mboxPage",this._);d.addParameter("screenHeight",this.ab);d.addParameter("screenWidth",this.bb);d.addParameter("browserWidth",this.cb);d.addParameter("browserHeight",this.db);d.addParameter("browserTimeOffset",this.fb);d.addParameter("colorDepth",this.eb);d.setUrlProcessAction(function(a){a+="&mboxURL="+encodeURIComponent(document.location);var b=encodeURIComponent(document.referrer);if(a.length+b.length<2000){a+="&mboxReferrer="+b}a+="&mboxVersion="+mboxVersion;return a})};mboxFactory.prototype.Ib=function(){return""};mboxFactory.prototype.ob=function(){document.write("<style>.mboxDefault { visibility:hidden; }</style>")};mboxFactory.prototype.isDomLoaded=function(){return this.T};mboxFactory.prototype.Z=function(){if(this.xb!=null){return}this.xb=new Array();var b=this;(function(){var h=document.addEventListener?"DOMContentLoaded":"onreadystatechange";var g=false;var f=function(){if(g){return}g=true;for(var c=0;c<b.xb.length;++c){b.xb[c]()}};if(document.addEventListener){document.addEventListener(h,function(){document.removeEventListener(h,arguments.callee,false);f()},false);window.addEventListener("load",function(){document.removeEventListener("load",arguments.callee,false);f()},false)}else{if(document.attachEvent){if(self!==self.top){document.attachEvent(h,function(){if(document.readyState==="complete"){document.detachEvent(h,arguments.callee);f()}})}else{var a=function(){try{document.documentElement.doScroll("left");f()}catch(c){setTimeout(a,13)}};a()}}}if(document.readyState==="complete"){f()}})()};mboxSignaler=function(i,l){this.Y=l;var h=l.getCookieNames("signal-");for(var j=0;j<h.length;j++){var n=h[j];var k=l.getCookie(n).split("&");var m=i(k[0],k);m.load();l.deleteCookie(n)}};mboxSignaler.prototype.signal=function(d,c){this.Y.setCookie("signal-"+d,mboxShiftArray(arguments).join("&"),45*60)};mboxList=function(){this.U=new Array()};mboxList.prototype.add=function(b){if(b!=null){this.U[this.U.length]=b}};mboxList.prototype.get=function(e){var f=new mboxList();for(var g=0;g<this.U.length;g++){var h=this.U[g];if(h.getName()==e){f.add(h)}}return f};mboxList.prototype.getById=function(b){return this.U[b]};mboxList.prototype.length=function(){return this.U.length};mboxList.prototype.each=function(d){if(typeof d!=="function"){throw"Action must be a function, was: "+typeof(d)}for(var c=0;c<this.U.length;c++){d(this.U[c])}};mboxLocatorDefault=function(b){this.w="mboxMarker-"+b;document.write('<div id="'+this.w+'" style="visibility:hidden;display:none">&nbsp;</div>')};mboxLocatorDefault.prototype.locate=function(){var b=document.getElementById(this.w);while(b!=null){if(b.nodeType==1){if(b.className=="mboxDefault"){return b}}b=b.previousSibling}return null};mboxLocatorDefault.prototype.force=function(){var d=document.createElement("div");d.className="mboxDefault";var c=document.getElementById(this.w);if(c){c.parentNode.insertBefore(d,c)}return d};mboxLocatorNode=function(b){this.Tb=b};mboxLocatorNode.prototype.locate=function(){return typeof this.Tb=="string"?document.getElementById(this.Tb):this.Tb};mboxLocatorNode.prototype.force=function(){return null};mboxCreate=function(d){var c=mboxFactoryDefault.create(d,mboxShiftArray(arguments));if(c){c.load()}return c};mboxDefine=function(f,e){var d=mboxFactoryDefault.create(e,mboxShiftArray(mboxShiftArray(arguments)),f);return d};mboxUpdate=function(b){mboxFactoryDefault.update(b,mboxShiftArray(arguments))};mbox=function(j,f,g,i,h){this.Zb=null;this._b=0;this.Cb=i;this.Db=h;this.ac=null;this.bc=new mboxOfferContent();this.Ub=null;this.L=g;this.message="";this.cc=new Object();this.dc=0;this.Xb=f;this.w=j;this.ec();g.addParameter("mbox",j).addParameter("mboxId",f);this.fc=function(){};this.gc=function(){};this.hc=null;this.ic=document.documentMode>=10&&!TNT.a.isDomLoaded;if(this.ic){this.jc=TNT.a.nestedMboxes;this.jc.push(this.w)}};mbox.prototype.getId=function(){return this.Xb};mbox.prototype.ec=function(){if(this.w.length>250){throw"Mbox Name "+this.w+" exceeds max length of 250 characters."}else{if(this.w.match(/^\s+|\s+$/g)){throw"Mbox Name "+this.w+" has leading/trailing whitespace(s)."}}};mbox.prototype.getName=function(){return this.w};mbox.prototype.getParameters=function(){var d=this.L.getParameters();var e=new Array();for(var f=0;f<d.length;f++){if(d[f].name.indexOf("mbox")!=0){e[e.length]=d[f].name+"="+d[f].value}}return e};mbox.prototype.setOnLoad=function(b){this.gc=b;return this};mbox.prototype.setMessage=function(b){this.message=b;return this};mbox.prototype.setOnError=function(b){this.fc=b;return this};mbox.prototype.setFetcher=function(b){if(this.ac){this.ac.cancel()}this.ac=b;return this};mbox.prototype.getFetcher=function(){return this.ac};mbox.prototype.load=function(d){if(this.ac==null){return this}this.setEventTime("load.start");this.cancelTimeout();this._b=0;var e=(d&&d.length>0)?this.L.clone().addParameters(d):this.L;this.ac.fetch(e);var f=this;this.lc=setTimeout(function(){f.fc("browser timeout",f.ac.getType())},15000);this.setEventTime("load.end");return this};mbox.prototype.loaded=function(){this.cancelTimeout();if(!this.activate()){var b=this;setTimeout(function(){b.loaded()},100)}};mbox.prototype.activate=function(){if(this._b){return this._b}this.setEventTime("activate"+ ++this.dc+".start");if(this.ic&&this.jc[this.jc.length-1]!==this.w){return this._b}if(this.show()){this.cancelTimeout();this._b=1}this.setEventTime("activate"+this.dc+".end");if(this.ic){this.jc.pop()}return this._b};mbox.prototype.isActivated=function(){return this._b};mbox.prototype.setOffer=function(b){if(b&&b.show&&b.setOnLoad){this.bc=b}else{throw"Invalid offer"}return this};mbox.prototype.getOffer=function(){return this.bc};mbox.prototype.show=function(){this.setEventTime("show.start");var b=this.bc.show(this);this.setEventTime(b==1?"show.end.ok":"show.end");return b};mbox.prototype.showContent=function(b){if(b==null){return 0}if(this.Ub==null||!this.Ub.parentNode){this.Ub=this.getDefaultDiv();if(this.Ub==null){return 0}}if(this.Ub!=b){this.nc(this.Ub);this.Ub.parentNode.replaceChild(b,this.Ub);this.Ub=b}this.oc(b);this.gc();return 1};mbox.prototype.hide=function(){this.setEventTime("hide.start");var b=this.showContent(this.getDefaultDiv());this.setEventTime(b==1?"hide.end.ok":"hide.end.fail");return b};mbox.prototype.finalize=function(){this.setEventTime("finalize.start");this.cancelTimeout();if(this.getDefaultDiv()==null){if(this.Cb.force()!=null){this.setMessage("No default content, an empty one has been added")}else{this.setMessage("Unable to locate mbox")}}if(!this.activate()){this.hide();this.setEventTime("finalize.end.hide")}this.setEventTime("finalize.end.ok")};mbox.prototype.cancelTimeout=function(){if(this.lc){clearTimeout(this.lc)}if(this.ac!=null){this.ac.cancel()}};mbox.prototype.getDiv=function(){return this.Ub};mbox.prototype.getDefaultDiv=function(){if(this.hc==null){this.hc=this.Cb.locate()}return this.hc};mbox.prototype.setEventTime=function(b){this.cc[b]=(new Date()).getTime()};mbox.prototype.getEventTimes=function(){return this.cc};mbox.prototype.getImportName=function(){return this.Db};mbox.prototype.getURL=function(){return this.L.buildUrl()};mbox.prototype.getUrlBuilder=function(){return this.L};mbox.prototype.qc=function(b){return b.style.display!="none"};mbox.prototype.oc=function(b){this.rc(b,true)};mbox.prototype.nc=function(b){this.rc(b,false)};mbox.prototype.rc=function(d,c){d.style.visibility=c?"visible":"hidden";d.style.display=c?"block":"none"};mbox.prototype.nb=function(){this.ic=false};mbox.prototype.relocateDefaultDiv=function(){this.hc=this.Cb.locate()};mboxOfferContent=function(){this.gc=function(){}};mboxOfferContent.prototype.show=function(c){var d=c.showContent(document.getElementById(c.getImportName()));if(d==1){this.gc()}return d};mboxOfferContent.prototype.setOnLoad=function(b){this.gc=b};mboxOfferAjax=function(b){this.mc=b;this.gc=function(){}};mboxOfferAjax.prototype.setOnLoad=function(b){this.gc=b};mboxOfferAjax.prototype.show=function(f){var e=document.createElement("div");e.id=f.getImportName();e.innerHTML=this.mc;var d=f.showContent(e);if(d==1){this.gc()}return d};mboxOfferDefault=function(){this.gc=function(){}};mboxOfferDefault.prototype.setOnLoad=function(b){this.gc=b};mboxOfferDefault.prototype.show=function(c){var d=c.hide();if(d==1){this.gc()}return d};mboxCookieManager=function mboxCookieManager(d,c){this.w=d;this.uc=c==""||c.indexOf(".")==-1?"":"; domain="+c;this.vc=new mboxMap();this.loadCookies()};mboxCookieManager.prototype.isEnabled=function(){this.setCookie("check","true",60);this.loadCookies();return this.getCookie("check")=="true"};mboxCookieManager.prototype.setCookie=function(e,f,g){if(typeof e!="undefined"&&typeof f!="undefined"&&typeof g!="undefined"){var h=new Object();h.name=e;h.value=escape(f);h.expireOn=Math.ceil(g+new Date().getTime()/1000);this.vc.put(e,h);this.saveCookies()}};mboxCookieManager.prototype.getCookie=function(d){var c=this.vc.get(d);return c?unescape(c.value):null};mboxCookieManager.prototype.deleteCookie=function(b){this.vc.remove(b);this.saveCookies()};mboxCookieManager.prototype.getCookieNames=function(d){var c=new Array();this.vc.each(function(b,a){if(b.indexOf(d)==0){c[c.length]=b}});return c};mboxCookieManager.prototype.saveCookies=function(){var f=false;var i="disable";var h=new Array();var g=0;this.vc.each(function(b,a){if(!f||b===i){h[h.length]=b+"#"+a.value+"#"+a.expireOn;if(g<a.expireOn){g=a.expireOn}}});var j=new Date(g*1000);document.cookie=this.w+"="+h.join("|")+"; expires="+j.toGMTString()+"; path=/"+this.uc};mboxCookieManager.prototype.loadCookies=function(){this.vc=new mboxMap();var n=document.cookie.indexOf(this.w+"=");if(n!=-1){var l=document.cookie.indexOf(";",n);if(l==-1){l=document.cookie.indexOf(",",n);if(l==-1){l=document.cookie.length}}var k=document.cookie.substring(n+this.w.length+1,l).split("|");var i=Math.ceil(new Date().getTime()/1000);for(var j=0;j<k.length;j++){var m=k[j].split("#");if(i<=m[2]){var h=new Object();h.name=m[0];h.value=m[1];h.expireOn=m[2];this.vc.put(h.name,h)}}}};mboxSession=function(g,f,i,j,h){this.Kc=f;this.Qb=i;this.Lc=j;this.Y=h;this.Mc=false;this.Xb=typeof mboxForceSessionId!="undefined"?mboxForceSessionId:mboxGetPageParameter(this.Kc);if(this.Xb==null||this.Xb.length==0){this.Xb=h.getCookie(i);if(this.Xb==null||this.Xb.length==0){this.Xb=g;this.Mc=true}}h.setCookie(i,this.Xb,j)};mboxSession.prototype.getId=function(){return this.Xb};mboxSession.prototype.forceId=function(b){this.Xb=b;this.Y.setCookie(this.Qb,this.Xb,this.Lc)};mboxPC=function(d,e,f){this.Qb=d;this.Lc=e;this.Y=f;this.Xb=typeof mboxForcePCId!="undefined"?mboxForcePCId:f.getCookie(d);if(this.Xb!=null){f.setCookie(d,this.Xb,e)}};mboxPC.prototype.getId=function(){return this.Xb};mboxPC.prototype.forceId=function(b){if(this.Xb!=b){this.Xb=b;this.Y.setCookie(this.Qb,this.Xb,this.Lc);return true}return false};mboxGetPageParameter=function(e){var f=null;var h=new RegExp("\\?[^#]*"+e+"=([^&;#]*)");var g=h.exec(document.location);if(g!=null&&g.length>=2){f=g[1]}return f};mboxSetCookie=function(d,e,f){return mboxFactoryDefault.getCookieManager().setCookie(d,e,f)};mboxGetCookie=function(b){return mboxFactoryDefault.getCookieManager().getCookie(b)};mboxCookiePageDomain=function(){var e=(/([^:]*)(:[0-9]{0,5})?/).exec(document.location.host)[1];var d=/[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}/;if(!d.exec(e)){var f=(/([^\.]+\.[^\.]{3}|[^\.]+\.[^\.]+\.[^\.]{2})$/).exec(e);if(f){e=f[0];if(e.indexOf("www.")==0){e=e.substr(4)}}}return e?e:""};mboxShiftArray=function(e){var d=new Array();for(var f=1;f<e.length;f++){d[d.length]=e[f]}return d};mboxGenerateId=function(){return(new Date()).getTime()+"-"+Math.floor(Math.random()*999999)};mboxScreenHeight=function(){return screen.height};mboxScreenWidth=function(){return screen.width};mboxBrowserWidth=function(){return(window.innerWidth)?window.innerWidth:document.documentElement?document.documentElement.clientWidth:document.body.clientWidth};mboxBrowserHeight=function(){return(window.innerHeight)?window.innerHeight:document.documentElement?document.documentElement.clientHeight:document.body.clientHeight};mboxBrowserTimeOffset=function(){return -new Date().getTimezoneOffset()};mboxScreenColorDepth=function(){return screen.pixelDepth};mboxBarclaysCookieConsent=function(){if(typeof _ccpCat3!="undefined"&&_ccpCat3===true){mboxFactoryDefault.enable(true)}else{mboxFactoryDefault.disable(60*60,"ccp")}};if(typeof mboxVersion=="undefined"){var mboxVersion=55;var mboxFactories=new mboxMap();var mboxFactoryDefault=new mboxFactory("barclaysbankplc.tt.omtrdc.net","barclaysbankplc","default");mboxBarclaysCookieConsent()}if(mboxGetPageParameter("mboxDebug")!=null||mboxFactoryDefault.getCookieManager().getCookie("debug")!=null){setTimeout(function(){if(typeof mboxDebugLoaded=="undefined"){alert("Could not load the remote debug.\nPlease check your connection to Test&amp;Target servers")}},60*60);document.write('<script language="Javascript1.2" src="//admin7.testandtarget.omniture.com/admin/mbox/mbox_debug.jsp?mboxServerHost=barclaysbankplc.tt.omtrdc.net&clientCode=barclaysbankplc"><\/script>')}mboxScPluginFetcher=function(c,d){this.r=c;this.Tc=d};mboxScPluginFetcher.prototype.Uc=function(d){d.setBasePath("/m2/"+this.r+"/sc/standard");this.Vc(d);var c=d.buildUrl();c+="&scPluginVersion=1";return c};mboxScPluginFetcher.prototype.Vc=function(e){var d=["dynamicVariablePrefix","visitorID","vmk","ppu","charSet","visitorNamespace","cookieDomainPeriods","cookieLifetime","pageName","currencyCode","variableProvider","channel","server","pageType","transactionID","purchaseID","campaign","state","zip","events","products","linkName","linkType","resolution","colorDepth","javascriptVersion","javaEnabled","cookiesEnabled","browserWidth","browserHeight","connectionType","homepage","pe","pev1","pev2","pev3","visitorSampling","visitorSamplingGroup","dynamicAccountSelection","dynamicAccountList","dynamicAccountMatch","trackDownloadLinks","trackExternalLinks","trackInlineStats","linkLeaveQueryString","linkDownloadFileTypes","linkExternalFilters","linkInternalFilters","linkTrackVars","linkTrackEvents","linkNames","lnk","eo"];for(var f=0;f<d.length;f++){this.Xc(d[f],e)}for(var f=1;f<=75;f++){this.Xc("prop"+f,e);this.Xc("eVar"+f,e);this.Xc("hier"+f,e)}};mboxScPluginFetcher.prototype.Xc=function(f,d){var e=this.Tc[f];if(typeof e==="undefined"||e===null||e===""||typeof e==="object"){return}d.addParameter(f,e)};mboxScPluginFetcher.prototype.cancel=function(){};mboxScPluginFetcher.prototype.fetch=function(d){d.setServerType(this.getType());var c=this.Uc(d);this.M=document.createElement("script");this.M.src=c;document.body.appendChild(this.M)};mboxScPluginFetcher.prototype.getType=function(){return"ajax"};function mboxLoadSCPlugin(b){if(!b){return null}b.m_tt=function(d){var a=d.m_i("tt");a.W=true;a.r="barclaysbankplc";a._t=function(){if(!this.isEnabled()){return}var c=this._c();if(c){var f=new mboxScPluginFetcher(this.r,this.s);c.setFetcher(f);c.load()}};a.isEnabled=function(){return this.W&&mboxFactoryDefault.isEnabled()};a._c=function(){var c=this.ad();var f=document.createElement("DIV");return mboxFactoryDefault.create(c,new Array(),f)};a.ad=function(){var c=this.s.events&&this.s.events.indexOf("purchase")!=-1;return"SiteCatalyst: "+(c?"purchase":"event")}};return b.loadModule("tt")}mboxVizTargetUrl=function(h){if(!mboxFactoryDefault.isEnabled()){return}var f=mboxFactoryDefault.getUrlBuilder().clone();f.setBasePath("/m2/barclaysbankplc/viztarget");f.addParameter("mbox",h);f.addParameter("mboxId",0);f.addParameter("mboxCount",mboxFactoryDefault.getMboxes().length()+1);var g=new Date();f.addParameter("mboxTime",g.getTime()-(g.getTimezoneOffset()*60000));f.addParameter("mboxPage",mboxGenerateId());var e=mboxShiftArray(arguments);if(e&&e.length>0){f.addParameters(e)}f.addParameter("mboxDOMLoaded",mboxFactoryDefault.isDomLoaded());mboxFactoryDefault.setVisitorIdParameters(f,h);return f.buildUrl()};TNT.createGlobalMbox=function(){var h="Global_Mbox";var g=("".length===0);var f="";var j;if(g){f="mbox-"+h+"-"+mboxGenerateId();j=document.createElement("div");j.className="mboxDefault";j.id=f;j.style.visibility="hidden";j.style.display="none";mboxFactoryDefault.addOnLoad(function(){document.body.insertBefore(j,document.body.firstChild)})}var i=mboxFactoryDefault.create(h,TNT.a.b(),f);if(i!=null){i.load()}};var mboxTrack=function(i,j){var h,f,l,k=mboxFactoryDefault;if(k.getMboxes().length()>0){h=k.getMboxes().getById(0);f=h.getURL().replace("mbox="+h.getName(),"mbox="+i).replace("mboxPage="+k.getPageId(),"mboxPage="+mboxGenerateId())+"&"+j,l=new Image();l.style.display="none";l.src=f;document.body.appendChild(l)}else{k.getSignaler().signal("onEvent",i+"&"+j)}},mboxTrackLink=function(f,d,e){mboxTrack(f,d);setTimeout("location='"+e+"'",500)};function tt_Log(b){mboxTrack("barc_onClick","Destination="+b)}function tt_Redirect(b){mboxTrack("barc_onClick","Destination="+b);setTimeout("location='"+b+"'",500)}mboxStandardFetcher.prototype.getType=function(){var a=this.x.attributes.src;if(a){a=a.value;if(a){if(a.indexOf("mbox=Global_Mbox")>-1){return"standard"}}}return"ajax"};if(typeof(mboxStandardScPluginFetcher)!="undefined"){mboxStandardScPluginFetcher.prototype.getType=function(){return"ajax"}}mboxStandardFetcher.prototype.fetch=function(g){for(var a=0,b=g.s.length;a<b;a++){if(g.s[a]["name"]=="mbox"){if(g.s[a]["value"]=="Global_Mbox"){g.setServerType("standard");document.write('<script src="'+g.buildUrl()+'" language="JavaScript"><\/script>')}else{g.setServerType("ajax");var h=g.buildUrl();this.x=document.createElement("script");this.x.src=h;document.getElementsByTagName("head")[0].appendChild(this.x)}break}}};if(typeof(mboxStandardScPluginFetcher)!="undefined"){mboxStandardScPluginFetcher.prototype.fetch=function(b){b.setServerType("ajax");var a=this._buildUrl(b);this._include=document.createElement("script");this._include.src=a;document.getElementsByTagName("head")[0].appendChild(this._include)}}var cmid=document.querySelector("meta[name='WT.dcsvid']");if(cmid){cmid=cmid.getAttribute("content")}mboxFactory.prototype.update=function(b,c){if(!this.isEnabled()){return}if(cmid){this.U.get(b).each(function(d){d.getUrlBuilder().addParameter("membershipid",cmid)})}var a=this;if(!this.isDomLoaded()){this.addOnLoad(function(){a.update(b,c)});return}if(this.U.get(b).length()==0){throw"Mbox "+b+" is not defined"}this.U.get(b).each(function(d){d.Rb=function(f,e){d.setMessage(f);d.activate();if(!d.isActivated()){a.disable(60*60,f)}};d.getUrlBuilder().addParameter("mboxPage",mboxGenerateId());mboxFactoryDefault.setVisitorIdParameters(d.getUrlBuilder(),b);d.load(c)})};mboxFactory.prototype.create=function(s,m,w){if(!this.isSupported()){return null}var n=this.L.clone();n.addParameter("mboxCount",this.U.length()+1);if(cmid){n.addParameter("membershipid",cmid)}n.addParameters(m);this.setVisitorIdParameters(n,s);var u=this.U.get(s).length();var q=this.S+"-"+s+"-"+u;var r;if(w){r=new mboxLocatorNode(w)}else{if(this.T){throw"The page has already been loaded, can't write marker"}r=new mboxLocatorDefault(q)}try{var v=this;var x="mboxImported-"+q;var p=new mbox(s,u,n,r,x);if(this.W){p.setFetcher(this.T?new mboxAjaxFetcher():new mboxStandardFetcher())}p.setOnError(function(a,b){p.setMessage(a);p.activate();if(!p.isActivated()){v.disable(60*60,a);window.location.reload(false)}});this.U.add(p)}catch(t){this.disable();throw'Failed creating mbox "'+s+'", the error was: '+t}var y=new Date();n.addParameter("mboxTime",y.getTime()-(y.getTimezoneOffset()*60000));return p};