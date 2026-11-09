/*
 * axionFEM viewer: interactive display of solved axisymmetric detector scenes
 * (schema "axionfem.scene/1", see docs/scene_format.md) as a 3D cutaway and as
 * a mirrored 2D meridional section.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, the axionFEM authors.
 *
 *   import { createViewer } from './viewer.js';
 *   const v = createViewer(document.getElementById('app'), scene, { mode: '3d' });
 *   v.setQuantity('Bz'); v.setScale('log'); v.setMode('section');
 *
 * The module imports 'three', 'three/addons/controls/OrbitControls.js' and
 * 'three/addons/environments/RoomEnvironment.js'; an import map must resolve
 * them (three.js r185, vendored in ../vendor). See docs/viewer.md.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

export const VERSION = '0.1.0';
export const SCHEMA = 'axionfem.scene/1';

const DEG = Math.PI / 180;
const MINUS = '−';
const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

/* ------------------------------------------------------------------ colormaps
 * 256-entry sRGB tables (hex RGB). viridis, inferno, magma, plasma, cividis:
 * the matplotlib maps (CC0; van der Walt & Smith, Nunez et al.). coolwarm:
 * Moreland's (2009) diverging map, as tabulated by matplotlib.            */
const LUT_HEX = {
  viridis: '44015444025645045745055946075a46085c460a5d460b5e470d60470e6147106347116447136548146748166848176948186a481a6c481b6d481c6e481d6f481f70482071482173482374482475482576482677482878482979472a7a472c7a472d7b472e7c472f7d46307e46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f854240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a3d4e8a3c4f8a3c508b3b518b3b528b3a538b3a548c39558c39568c38588c38598c375a8c375b8d365c8d365d8d355e8d355f8d34608d34618d33628d33638d32648e32658e31668e31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e277e8e277f8e27808e26818e26828e26828e25838e25848e25858e24868e24878e23888e23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d21918c20928c20928c20938c1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f881fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa8325ab8225ac8226ad8127ad8128ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b32b67a34b67935b77937b87838b9773aba763bbb753dbc743fbc7340bd7242be7144bf7046c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc8635ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d1537ad1517cd2507fd34e81d34d84d44b86d54989d5488bd6468ed64590d74393d74195d84098d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32addc30b0dd2fb2dd2db5de2bb8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21ad8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61ef6e620f8e621fbe723fde725',
  inferno: '00000401000501010601010802010a02020c02020e03021004031204031405041706041907051b08051d09061f0a07220b07240c08260d08290e092b10092d110a30120a32140b34150b37160b39180c3c190c3e1b0c411c0c431e0c451f0c48210c4a230c4c240c4f260c51280b53290b552b0b572d0b592f0a5b310a5c320a5e340a5f3609613809623909633b09643d09653e0966400a67420a68440a68450a69470b6a490b6a4a0c6b4c0c6b4d0d6c4f0d6c510e6c520e6d540f6d550f6d57106e59106e5a116e5c126e5d126e5f136e61136e62146e64156e65156e67166e69166e6a176e6c186e6d186e6f196e71196e721a6e741a6e751b6e771c6d781c6d7a1d6d7c1d6d7d1e6d7f1e6c801f6c82206c84206b85216b87216b88226a8a226a8c23698d23698f24699025689225689326679526679727669827669a28659b29649d29649f2a63a02a63a22b62a32c61a52c60a62d60a82e5fa92e5eab2f5ead305dae305cb0315bb1325ab3325ab43359b63458b73557b93556ba3655bc3754bd3853bf3952c03a51c13a50c33b4fc43c4ec63d4dc73e4cc83f4bca404acb4149cc4248ce4347cf4446d04545d24644d34743d44842d54a41d74b3fd84c3ed94d3dda4e3cdb503bdd513ade5238df5337e05536e15635e25734e35933e45a31e55c30e65d2fe75e2ee8602de9612bea632aeb6429eb6628ec6726ed6925ee6a24ef6c23ef6e21f06f20f1711ff1731df2741cf3761bf37819f47918f57b17f57d15f67e14f68013f78212f78410f8850ff8870ef8890cf98b0bf98c0af98e09fa9008fa9207fa9407fb9606fb9706fb9906fb9b06fb9d07fc9f07fca108fca309fca50afca60cfca80dfcaa0ffcac11fcae12fcb014fcb216fcb418fbb61afbb81dfbba1ffbbc21fbbe23fac026fac228fac42afac62df9c72ff9c932f9cb35f8cd37f8cf3af7d13df7d340f6d543f6d746f5d949f5db4cf4dd4ff4df53f4e156f3e35af3e55df2e661f2e865f2ea69f1ec6df1ed71f1ef75f1f179f2f27df2f482f3f586f3f68af4f88ef5f992f6fa96f8fb9af9fc9dfafda1fcffa4',
  magma: '00000401000501010601010802010902020b02020d03030f03031204041405041606051806051a07061c08071e0907200a08220b09240c09260d0a290e0b2b100b2d110c2f120d31130d34140e36150e38160f3b180f3d19103f1a10421c10441d11471e114920114b21114e22115024125325125527125829115a2a115c2c115f2d11612f116331116533106734106936106b38106c390f6e3b0f703d0f713f0f72400f74420f75440f764510774710784910784a10794c117a4e117b4f127b51127c52137c54137d56147d57157e59157e5a167e5c167f5d177f5f187f601880621980641a80651a80671b80681c816a1c816b1d816d1d816e1e81701f81721f817320817521817621817822817922827b23827c23827e24828025828125818326818426818627818827818928818b29818c29818e2a81902a81912b81932b80942c80962c80982d80992d809b2e7f9c2e7f9e2f7fa02f7fa1307ea3307ea5317ea6317da8327daa337dab337cad347cae347bb0357bb2357bb3367ab5367ab73779b83779ba3878bc3978bd3977bf3a77c03a76c23b75c43c75c53c74c73d73c83e73ca3e72cc3f71cd4071cf4070d0416fd2426fd3436ed5446dd6456cd8456cd9466bdb476adc4869de4968df4a68e04c67e24d66e34e65e44f64e55064e75263e85362e95462ea5661eb5760ec5860ed5a5fee5b5eef5d5ef05f5ef1605df2625df2645cf3655cf4675cf4695cf56b5cf66c5cf66e5cf7705cf7725cf8745cf8765cf9785df9795df97b5dfa7d5efa7f5efa815ffb835ffb8560fb8761fc8961fc8a62fc8c63fc8e64fc9065fd9266fd9467fd9668fd9869fd9a6afd9b6bfe9d6cfe9f6dfea16efea36ffea571fea772fea973feaa74feac76feae77feb078feb27afeb47bfeb67cfeb77efeb97ffebb81febd82febf84fec185fec287fec488fec68afec88cfeca8dfecc8ffecd90fecf92fed194fed395fed597fed799fed89afdda9cfddc9efddea0fde0a1fde2a3fde3a5fde5a7fde7a9fde9aafdebacfcecaefceeb0fcf0b2fcf2b4fcf4b6fcf6b8fcf7b9fcf9bbfcfbbdfcfdbf',
  plasma: '0d088710078813078916078a19068c1b068d1d068e20068f2206902406912605912805922a05932c05942e05952f059631059733059735049837049938049a3a049a3c049b3e049c3f049c41049d43039e44039e46039f48039f4903a04b03a14c02a14e02a25002a25102a35302a35502a45601a45801a45901a55b01a55c01a65e01a66001a66100a76300a76400a76600a76700a86900a86a00a86c00a86e00a86f00a87100a87201a87401a87501a87701a87801a87a02a87b02a87d03a87e03a88004a88104a78305a78405a78606a68707a68808a68a09a58b0aa58d0ba58e0ca48f0da4910ea3920fa39410a29511a19613a19814a099159f9a169f9c179e9d189d9e199da01a9ca11b9ba21d9aa31e9aa51f99a62098a72197a82296aa2395ab2494ac2694ad2793ae2892b02991b12a90b22b8fb32c8eb42e8db52f8cb6308bb7318ab83289ba3388bb3488bc3587bd3786be3885bf3984c03a83c13b82c23c81c33d80c43e7fc5407ec6417dc7427cc8437bc9447aca457acb4679cc4778cc4977cd4a76ce4b75cf4c74d04d73d14e72d24f71d35171d45270d5536fd5546ed6556dd7566cd8576bd9586ada5a6ada5b69db5c68dc5d67dd5e66de5f65de6164df6263e06363e16462e26561e26660e3685fe4695ee56a5de56b5de66c5ce76e5be76f5ae87059e97158e97257ea7457eb7556eb7655ec7754ed7953ed7a52ee7b51ef7c51ef7e50f07f4ff0804ef1814df1834cf2844bf3854bf3874af48849f48948f58b47f58c46f68d45f68f44f79044f79143f79342f89441f89540f9973ff9983ef99a3efa9b3dfa9c3cfa9e3bfb9f3afba139fba238fca338fca537fca636fca835fca934fdab33fdac33fdae32fdaf31fdb130fdb22ffdb42ffdb52efeb72dfeb82cfeba2cfebb2bfebd2afebe2afec029fdc229fdc328fdc527fdc627fdc827fdca26fdcb26fccd25fcce25fcd025fcd225fbd324fbd524fbd724fad824fada24f9dc24f9dd25f8df25f8e125f7e225f7e425f6e626f6e826f5e926f5eb27f4ed27f3ee27f3f027f2f227f1f426f1f525f0f724f0f921',
  cividis: '00224e00234f00245100255300255400265600275800285900285b00295d002a5f002a61002b62002c64002c66002d68002e6a002e6c002f6d00306f0030700031700031710132710533710833700c34700f357012357014367016377018376f1a386f1c396f1e3a6f203a6f213b6e233c6e243c6e263d6e273e6e293f6e2a3f6d2b406d2d416d2e416d2f426d31436d32436d33446d34456c35456c36466c38476c39486c3a486c3b496c3c4a6c3d4a6c3e4b6c3f4c6c404c6c414d6c424e6c434e6c444f6c45506c46516c47516c48526c49536c4a536c4b546c4c556c4d556c4e566c4f576c50576c51586d52596d535a6d545a6d555b6d555c6d565c6d575d6d585e6d595e6e5a5f6e5b606e5c616e5d616e5e626e5e636f5f636f60646f61656f62656f636670646770656870656870666970676a71686a71696b716a6c716b6d726c6d726c6e726d6f726e6f736f70737071737172747272747273747374757474757575757676767777767777777878777979777a7a787b7a787c7b787d7c787e7c787e7d787f7e78807f78817f788280798381798482798582798683798784788885788985788a86788b87788c88788d88788e89788f8a78908b78918b78928c78928d78938e78948e77958f779690779791779892779992779a93769b94769c95769d95769e96769f9775a09875a19975a29975a39a74a49b74a59c74a69c74a79d73a89e73a99f73aaa073aba072aca172ada272aea371afa471b0a571b1a570b3a670b4a76fb5a86fb6a96fb7a96eb8aa6eb9ab6dbaac6dbbad6dbcae6cbdae6cbeaf6bbfb06bc0b16ac1b26ac2b369c3b369c4b468c5b568c6b667c7b767c8b866c9b965cbb965ccba64cdbb63cebc63cfbd62d0be62d1bf61d2c060d3c05fd4c15fd5c25ed6c35dd7c45cd9c55cdac65bdbc75adcc859ddc858dec958dfca57e0cb56e1cc55e2cd54e4ce53e5cf52e6d051e7d150e8d24fe9d34eead34cebd44bedd54aeed649efd748f0d846f1d945f2da44f3db42f5dc41f6dd3ff7de3ef8df3cf9e03afbe138fce236fde334fee434fee535fee636fee838',
  coolwarm: '3b4cc03c4ec23d50c33e51c53f53c64055c84257c94358cb445acc455cce465ecf485fd14961d24a63d34b64d54c66d64e68d84f69d9506bda516ddb536edd5470de5572df5673e05875e15977e35a78e45b7ae55d7ce65e7de75f7fe86180e96282ea6384eb6485ec6687ed6788ee688aef6a8bef6b8df06c8ff16e90f26f92f37093f37295f47396f57597f67699f6779af7799cf87a9df87b9ff97da0f97ea1fa80a3fa81a4fb82a6fb84a7fc85a8fc86a9fc88abfd89acfd8badfd8caffe8db0fe8fb1fe90b2fe92b4fe93b5fe94b6ff96b7ff97b8ff98b9ff9abbff9bbcff9dbdff9ebeff9fbfffa1c0ffa2c1ffa3c2fea5c3fea6c4fea7c5fea9c6fdaac7fdabc8fdadc9fdaec9fcafcafcb1cbfcb2ccfbb3cdfbb5cdfab6cefab7cff9b9d0f9bad0f8bbd1f8bcd2f7bed2f6bfd3f6c0d4f5c1d4f4c3d5f4c4d5f3c5d6f2c6d6f1c7d7f0c9d7f0cad8efcbd8eeccd9edcdd9eccedaebcfdaead1dae9d2dbe8d3dbe7d4dbe6d5dbe5d6dce4d7dce3d8dce2d9dce1dadce0dbdcdedcdddddddcdcdedcdbdfdbd9e0dbd8e1dad6e2dad5e3d9d3e4d9d2e5d8d1e6d7cfe7d7cee8d6cce9d5cbead5c9ead4c8ebd3c6ecd3c5edd2c3edd1c2eed0c0efcfbfefcebdf0cdbbf1cdbaf1ccb8f2cbb7f2cab5f2c9b4f3c8b2f3c7b1f4c6aff4c5adf5c4acf5c2aaf5c1a9f5c0a7f6bfa6f6bea4f6bda2f7bca1f7ba9ff7b99ef7b89cf7b79bf7b599f7b497f7b396f7b194f7b093f7af91f7ad90f7ac8ef7aa8cf7a98bf7a889f7a688f6a586f6a385f6a283f5a081f59f80f59d7ef59c7df49a7bf4987af39778f39577f39475f29274f29072f18f71f18d6ff08b6ef08a6cef886bee8669ee8468ed8366ec8165ec7f63eb7d62ea7b60e97a5fe9785de8765ce7745be67259e57058e46e56e36c55e36b54e26952e16751e0654fdf634ede614ddd5f4bdc5d4ada5a49d95847d85646d75445d65244d55042d44e41d24b40d1493fd0473dcf453ccd423bcc403acb3e38ca3b37c83836c73635c53334c43032c32e31c12b30c0282fbe242ebd1f2dbb1b2cba162bb8122ab70d28b50927b40426',
};
export const COLORMAPS = Object.keys(LUT_HEX);
const LUT_CACHE = {};
function lutBytes(name) {
  const key = LUT_HEX[name] ? name : 'viridis';
  if (!LUT_CACHE[key]) {
    const s = LUT_HEX[key];
    const a = new Uint8Array(768);
    for (let i = 0; i < 768; i++) a[i] = parseInt(s.substr(2 * i, 2), 16);
    LUT_CACHE[key] = a;
  }
  return LUT_CACHE[key];
}

/* ------------------------------------------------------- quantities and roles */
const QMETA = {
  Bmag: { sym: [['|B|']], text: '|B|', name: 'static field magnitude', family: 'static', kind: 'magnitude',
    desc: 'Magnitude of the static laboratory field B_lab.' },
  Br: { sym: [['B'], ['r', 'sub']], text: 'B_r', name: 'static field, radial', family: 'static', kind: 'component',
    desc: 'Radial component of the static laboratory field B_lab.' },
  Bz: { sym: [['B'], ['z', 'sub']], text: 'B_z', name: 'static field, axial', family: 'static', kind: 'component',
    desc: 'Axial component of the static laboratory field B_lab.' },
  Bphi: { sym: [['B'], ['φ', 'sub']], text: 'B_φ', name: 'static field, azimuthal', family: 'static', kind: 'component',
    desc: 'Azimuthal (toroidal) component of the static laboratory field B_lab.' },
  psi: { sym: [['ψ = rA'], ['φ', 'sub']], text: 'ψ = rA_φ', name: 'static flux function', family: 'static', kind: 'flux',
    desc: 'Flux function ψ = r A_φ of B_lab (2πψ is the flux through the circle of radius r). Its contours are the exact poloidal field lines.' },
  pickup_Bmag: { sym: [['|B'], ['I', 'sub'], ['|']], text: '|B_I|', name: 'pickup-mode field per ampere', family: 'pickup', kind: 'magnitude',
    desc: 'Field of a unit readout current in the pickup (per ampere).' },
  pickup_psi: { sym: [['ψ'], ['I', 'sub']], text: 'ψ_I', name: 'pickup flux function per ampere', family: 'pickup', kind: 'flux',
    desc: 'Flux function r A_φ of the pickup mode per ampere; its contours are the pickup field lines.' },
};
QMETA.Bx = { sym: [['B'], ['x', 'sub']], text: 'B_x', name: 'static field, x', family: 'static', kind: 'component', desc: 'x component of the static laboratory field B_lab.' };
QMETA.By = { sym: [['B'], ['y', 'sub']], text: 'B_y', name: 'static field, y', family: 'static', kind: 'component', desc: 'y component of the static laboratory field B_lab.' };
const QORDER = ['Bmag', 'Br', 'Bx', 'By', 'Bz', 'Bphi', 'psi', 'pickup_Bmag', 'pickup_psi'];
function qmeta(q) {
  return QMETA[q] || { sym: [[q]], text: q, name: q, family: /^pickup/.test(q) ? 'pickup' : 'static', kind: 'component', desc: '' };
}
const DEFAULT_UNITS = { Bmag: 'T', Br: 'T', Bx: 'T', By: 'T', Bz: 'T', Bphi: 'T', psi: 'Wb/rad', pickup_Bmag: 'T/A', pickup_psi: 'Wb/(rad A)' };

const ROLE = {
  superconductor: { label: 'superconductor', color: '#b3bbc6', flat: '#8f99a7', metal: 0.92, rough: 0.28, alpha2d: 1 },
  winding: { label: 'winding', color: '#c46a2c', metal: 0.85, rough: 0.34, alpha2d: 0.5 },
  conductor: { label: 'normal conductor', color: '#b8862b', metal: 0.85, rough: 0.32, alpha2d: 0.5 },
  iron: { label: 'iron', color: '#50555d', metal: 0.6, rough: 0.5, alpha2d: 0.92 },
  port: { label: 'readout port', color: '#2f9e6f', metal: 0.0, rough: 0.5, alpha2d: 0.6 },
  magnetic: { label: 'magnetic material', color: '#7b6aa8', metal: 0.2, rough: 0.6, alpha2d: 0.6 },
  vacuum: { label: 'vacuum', color: '#9fb7d6', metal: 0.0, rough: 0.8, alpha2d: 0.25, opacity: 0.25 },
};
const roleStyle = (role) => ROLE[role] || ROLE.vacuum;
const LINE_COLORS = { static: '#ffffff', pickup: '#6ff0ff' };
// field lines switch between a light and a dark variant according to the
// luminance of the colormap underneath, so they stay visible everywhere
const LINE_TONES = { static: ['#ffffff', '#0e1116'], pickup: ['#7ff3ff', '#003f4f'] };

const THEMES = {
  dark: { bg: '#0d1117', bgInner: '#1a2130', fg: '#e6edf3', fg2: '#9da7b3', axis: '#c9d1d9', halo: 'rgba(13,17,23,0.82)', outline: '#0b0e14' },
  light: { bg: '#e8ecf1', bgInner: '#ffffff', fg: '#1f2328', fg2: '#57606a', axis: '#3b434d', halo: 'rgba(250,251,252,0.88)', outline: '#0b0e14' },
};

/* ------------------------------------------------------------------ helpers */
function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else if (k === 'html') el.innerHTML = v;
      else el.setAttribute(k, v === true ? '' : String(v));
    }
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : String(c));
  }
  return el;
}
const lineSw = (color) => h('span', { class: 'axf-sw line', style: { backgroundImage: `linear-gradient(${color},${color})` } });
const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ESC[c]);
const SUPS = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
const sup = (n) => String(n).split('').map((c) => SUPS[c] ?? c).join('');
const minus = (s) => s.replace(/^-/, MINUS).replace(/e-/, 'e' + MINUS);

/** Number with `sig` significant figures; scientific (a×10ⁿ) outside [1e-3, 1e5). */
export function fmtNum(x, sig = 4) {
  if (x == null || typeof x !== 'number' || Number.isNaN(x)) return '—';
  if (!Number.isFinite(x)) return x > 0 ? '∞' : MINUS + '∞';
  if (x === 0) return '0';
  const ax = Math.abs(x);
  if (ax >= 1e-3 && ax < 1e5) {
    const s = x.toPrecision(sig);
    if (!s.includes('e')) return minus(s);
  }
  let e = Math.floor(Math.log10(ax));
  let m = (x / 10 ** e).toPrecision(sig);
  if (Math.abs(parseFloat(m)) >= 10) { e += 1; m = (x / 10 ** e).toPrecision(sig); }
  return `${minus(m)}×10${sup(e)}`;
}
const PREFIXES = { '-15': 'f', '-12': 'p', '-9': 'n', '-6': 'µ', '-3': 'm', '0': '', '3': 'k', '6': 'M', '9': 'G', '12': 'T' };
/** Number with an SI prefix on `unit` ("1.486 µH"). */
export function fmtSI(x, unit, sig = 4) {
  if (x == null || typeof x !== 'number' || !Number.isFinite(x)) return `${fmtNum(x)} ${unit}`;
  if (x === 0) return `0 ${unit}`;
  if (Math.abs(x) < 1e-12 || Math.abs(x) >= 1e15) return `${fmtNum(x, sig)} ${unit}`;
  let e3 = clamp(Math.floor(Math.log10(Math.abs(x)) / 3) * 3, -12, 12);
  let m = (x / 10 ** e3).toPrecision(sig);
  if (Math.abs(parseFloat(m)) >= 1000 && e3 < 12) { e3 += 3; m = (x / 10 ** e3).toPrecision(sig); }
  if (m.includes('e')) return `${fmtNum(x, sig)} ${unit}`;
  return `${minus(m)} ${PREFIXES[e3]}${unit}`;
}
/** Field-like value: plain in [0.01, 1e4), otherwise with an SI prefix. */
function fmtVal(v, unit, sig = 4) {
  if (v == null || !Number.isFinite(v)) return `${fmtNum(v)} ${unit}`;
  const a = Math.abs(v);
  if (a === 0) return `0 ${unit}`;
  if (a >= 1e-2 && a < 1e4) return `${minus(v.toPrecision(sig))} ${unit}`;
  return fmtSI(v, unit, sig);
}
function niceStep(span, n) {
  const raw = span / Math.max(1, n);
  const e = Math.floor(Math.log10(raw));
  const f = raw / 10 ** e;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * 10 ** e;
}
function niceTicks(a, b, n) {
  if (!(b > a)) return [a];
  const st = niceStep(b - a, n);
  const out = [];
  for (let t = Math.ceil(a / st - 1e-9) * st; t <= b + st * 1e-9; t += st) out.push(Math.abs(t) < st * 1e-9 ? 0 : t);
  return { ticks: out, step: st };
}
function tickText(t, step) {
  const d = clamp(-Math.floor(Math.log10(step) + 1e-9), 0, 12);
  return minus(t.toFixed(d));
}
function richHTML(parts) {
  return parts.map(([t, k]) => (k ? `<${k}>${esc(t)}</${k}>` : esc(t))).join('');
}
function drawRich(ctx, parts, x, y, size, weight = 600) {
  let cx = x;
  for (const [t, k] of parts) {
    const s = k ? size * 0.74 : size;
    ctx.font = `${weight} ${s}px ${FONT}`;
    const dy = k === 'sub' ? size * 0.3 : k === 'sup' ? -size * 0.4 : 0;
    ctx.fillText(t, cx, y + dy);
    cx += ctx.measureText(t).width;
  }
  return cx - x;
}
function measureRich(ctx, parts, size, weight = 600) {
  let w = 0;
  for (const [t, k] of parts) { ctx.font = `${weight} ${k ? size * 0.74 : size}px ${FONT}`; w += ctx.measureText(t).width; }
  return w;
}
function hexToRgb(hex) {
  const c = new THREE.Color(hex);
  return [c.r, c.g, c.b];
}
function mixHex(a, b, t) {
  return '#' + new THREE.Color(a).lerp(new THREE.Color(b), t).getHexString();
}
function prefixFor(maxAbs) {
  if (!(maxAbs > 0) || (maxAbs >= 1e-2 && maxAbs < 1e4)) return { e3: 0, p: '' };
  const e3 = clamp(Math.floor(Math.log10(maxAbs) / 3) * 3, -15, 12);
  return { e3, p: PREFIXES[e3] };
}

/* ------------------------------------------------------------------ decoding */
const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
const DTYPES = { float32: [Float32Array, 4], float64: [Float64Array, 8], uint32: [Uint32Array, 4], int32: [Int32Array, 4], uint16: [Uint16Array, 2], uint8: [Uint8Array, 1] };

/**
 * Decode a scene typed array {dtype, shape, data: base64 little-endian}.
 * Also accepts nested JS arrays and typed arrays. Returns {data, rows, cols}.
 */
export function decodeArray(a, what = 'array') {
  if (a == null) return { data: new Float32Array(0), rows: 0, cols: 0 };
  if (ArrayBuffer.isView(a)) return { data: a, rows: a.length, cols: 1 };
  if (Array.isArray(a)) {
    const rows = a.length;
    const cols = rows && Array.isArray(a[0]) ? a[0].length : 1;
    const out = new Float64Array(rows * cols);
    let k = 0;
    for (const row of a) {
      if (Array.isArray(row)) for (let j = 0; j < cols; j++) out[k++] = +row[j];
      else out[k++] = +row;
    }
    return { data: out, rows, cols };
  }
  if (typeof a !== 'object' || typeof a.data !== 'string') {
    throw new Error(`scene ${what}: expected a typed array {dtype, shape, data: base64}`);
  }
  const spec = DTYPES[a.dtype];
  if (!spec) throw new Error(`scene ${what}: unsupported dtype '${a.dtype}' (use float32 or uint32)`);
  const [Ctor, size] = spec;
  let bin;
  try { bin = atob(a.data); } catch (e) { throw new Error(`scene ${what}: data is not valid base64`); }
  if (bin.length % size) throw new Error(`scene ${what}: ${bin.length} bytes is not a multiple of ${size} (${a.dtype})`);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  if (!LITTLE_ENDIAN && size > 1) for (let i = 0; i < bytes.length; i += size) bytes.subarray(i, i + size).reverse();
  const data = new Ctor(bytes.buffer);
  const shape = Array.isArray(a.shape) ? a.shape : [];
  const cols = Number.isFinite(shape[1]) ? shape[1] : 1;
  const rows = Number.isFinite(shape[0]) ? shape[0] : data.length / Math.max(cols, 1);
  if (rows * cols !== data.length) {
    throw new Error(`scene ${what}: shape [${shape.join(', ')}] needs ${rows * cols} values but the data hold ${data.length}`);
  }
  return { data, rows, cols };
}
const toF32 = (a) => (a instanceof Float32Array ? a : Float32Array.from(a));
const toU32 = (a) => (a instanceof Uint32Array ? a : Uint32Array.from(a));

/* -------------------------------------------------------- outline processing */
function signedArea(P) {
  let s = 0;
  const n = P.length >> 1;
  for (let i = 0, j = n - 1; i < n; j = i++) s += P[2 * j] * P[2 * i + 1] - P[2 * i] * P[2 * j + 1];
  return s / 2;
}
function pointInPoly(P, x, y) {
  let inside = false;
  const n = P.length >> 1;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = P[2 * i], yi = P[2 * i + 1], xj = P[2 * j], yj = P[2 * j + 1];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function segDist2(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const l2 = dx * dx + dy * dy;
  let t = l2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0;
  t = clamp(t, 0, 1);
  const qx = ax + t * dx - px, qy = ay + t * dy - py;
  return qx * qx + qy * qy;
}
/** Douglas-Peucker simplification of a closed polyline (flat [x0,y0,x1,y1,...]). */
function simplifyClosed(P, tol) {
  const n = P.length >> 1;
  if (n <= 4) return P;
  let f = 0, best = -1;
  for (let i = 1; i < n; i++) {
    const d = (P[2 * i] - P[0]) ** 2 + (P[2 * i + 1] - P[1]) ** 2;
    if (d > best) { best = d; f = i; }
  }
  const keep = new Uint8Array(n);
  keep[0] = 1; keep[f] = 1;
  const tol2 = tol * tol;
  const stack = [[0, f], [f, n]];
  while (stack.length) {
    const [a, b] = stack.pop();
    if (b - a < 2) continue;
    const ia = a % n, ib = b % n;
    const ax = P[2 * ia], ay = P[2 * ia + 1], bx = P[2 * ib], by = P[2 * ib + 1];
    let md = -1, mi = -1;
    for (let k = a + 1; k < b; k++) {
      const d = segDist2(P[2 * k], P[2 * k + 1], ax, ay, bx, by);
      if (d > md) { md = d; mi = k; }
    }
    if (md > tol2) { keep[mi] = 1; stack.push([a, mi], [mi, b]); }
  }
  const out = [];
  for (let i = 0; i < n; i++) if (keep[i]) out.push(P[2 * i], P[2 * i + 1]);
  return out;
}
/**
 * Clean, simplify and orient boundary loops so that the material lies to the
 * left of the direction of travel (outer loops counter-clockwise in (r, z),
 * holes clockwise). The outward normal of an edge (dr, dz) is then (dz, -dr).
 */
function processLoops(raw, L) {
  const eps = 1e-9 * L, tol = 1.5e-4 * L;
  const loops = [];
  for (const a of raw) {
    const pts = [];
    for (let i = 0; i + 1 < a.length; i += 2) {
      const x = a[i], y = a[i + 1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const k = pts.length;
      if (k >= 2 && Math.abs(x - pts[k - 2]) <= eps && Math.abs(y - pts[k - 1]) <= eps) continue;
      pts.push(x, y);
    }
    while (pts.length >= 4 && Math.abs(pts[0] - pts[pts.length - 2]) <= eps && Math.abs(pts[1] - pts[pts.length - 1]) <= eps) pts.length -= 2;
    if (pts.length < 6) continue;
    loops.push({ pts: Float64Array.from(simplifyClosed(pts, tol)) });
  }
  for (const lp of loops) {
    let depth = 0;
    for (const other of loops) if (other !== lp && pointInPoly(other.pts, lp.pts[0], lp.pts[1])) depth++;
    lp.hole = depth % 2 === 1;
    const area = signedArea(lp.pts);
    if ((area > 0) === lp.hole) {
      const P = lp.pts, n = P.length >> 1, Q = new Float64Array(P.length);
      for (let i = 0; i < n; i++) { Q[2 * i] = P[2 * (n - 1 - i)]; Q[2 * i + 1] = P[2 * (n - 1 - i) + 1]; }
      lp.pts = Q;
    }
  }
  return loops;
}

/* -------------------------------------------------------------- scene model */
function bboxOf(V, b) {
  for (let i = 0; i + 1 < V.length; i += 2) {
    const x = V[i], y = V[i + 1];
    if (x < b[0]) b[0] = x; if (x > b[2]) b[2] = x;
    if (y < b[1]) b[1] = y; if (y > b[3]) b[3] = y;
  }
  return b;
}
function checkIndices(T, nv, what) {
  if (T.length % 3) throw new Error(`scene ${what}: index count ${T.length} is not a multiple of 3`);
  let mx = 0;
  for (let i = 0; i < T.length; i++) if (T[i] > mx) mx = T[i];
  if (T.length && mx >= nv) throw new Error(`scene ${what}: index ${mx} out of range (only ${nv} vertices)`);
}

/** Decode and pre-process an axionfem.scene/1 object into the viewer's model. */
export function prepareScene(scene) {
  if (!scene || typeof scene !== 'object') throw new Error('axionFEM viewer: the scene must be a parsed axionfem.scene/1 JSON object');
  const warnings = [];
  if (scene.schema && !String(scene.schema).startsWith('axionfem.scene/')) warnings.push(`unexpected schema '${scene.schema}'`);
  if (scene.symmetry && scene.symmetry !== 'axisymmetric') warnings.push(`symmetry '${scene.symmetry}' is drawn as axisymmetric`);
  const bb = [Infinity, Infinity, -Infinity, -Infinity];
  const regions = (scene.regions || []).map((r, i) => {
    const w = `regions[${i}] ('${r.name}')`;
    const V = toF32(decodeArray(r.triangles?.vertices, `${w}.triangles.vertices`).data);
    const T = toU32(decodeArray(r.triangles?.indices, `${w}.triangles.indices`).data);
    checkIndices(T, V.length >> 1, `${w}.triangles.indices`);
    const raw = (r.outline || []).map((o, k) => decodeArray(o, `${w}.outline[${k}]`).data);
    bboxOf(V, bb);
    for (const a of raw) bboxOf(a, bb);
    const role = ROLE[r.role] ? r.role : 'vacuum';
    return { index: i, name: String(r.name ?? `region ${i}`), role, material: r.material || '', color: r.color || roleStyle(role).color, V, T, raw };
  });
  const curves = (scene.curves || []).map((c, i) => {
    const S = toF32(decodeArray(c.segments, `curves[${i}].segments`).data);
    bboxOf(S, bb);
    return { name: String(c.name ?? `curve ${i}`), role: c.role || 'superconductor', color: c.color || ROLE.superconductor.color, S };
  });
  // field
  let field = null;
  const F = scene.field;
  if (F && F.vertices && F.indices) {
    const V = toF32(decodeArray(F.vertices, 'field.vertices').data);
    const T = toU32(decodeArray(F.indices, 'field.indices').data);
    const nv = V.length >> 1;
    checkIndices(T, nv, 'field.indices');
    const q = {}, stats = {};
    const names = Object.keys(F.quantities || {}).sort((a, b) => {
      const ia = QORDER.indexOf(a), ib = QORDER.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
    });
    for (const k of names) {
      const d = decodeArray(F.quantities[k], `field.quantities.${k}`).data;
      if (d.length !== nv) { warnings.push(`field quantity '${k}' has ${d.length} values for ${nv} vertices; skipped`); continue; }
      const a = toF32(d);
      let mn = Infinity, mx = -Infinity, maxAbs = 0, minPos = Infinity, nonfinite = 0;
      for (let i = 0; i < a.length; i++) {
        const v = a[i];
        if (!Number.isFinite(v)) { nonfinite++; continue; }
        if (v < mn) mn = v; if (v > mx) mx = v;
        const av = Math.abs(v);
        if (av > maxAbs) maxAbs = av;
        if (av > 0 && av < minPos) minPos = av;
      }
      if (nonfinite) warnings.push(`field quantity '${k}': ${nonfinite} non-finite values ignored`);
      if (!(mx >= mn)) { mn = 0; mx = 0; }
      q[k] = a;
      stats[k] = { min: mn, max: mx, maxAbs, minPos: Number.isFinite(minPos) ? minPos : 0, zero: maxAbs === 0, signed: mn < -1e-6 * maxAbs && mx > 1e-6 * maxAbs };
    }
    const units = Object.assign({}, DEFAULT_UNITS, F.units || {});
    field = { V, T, q, stats, units, names: Object.keys(q), bbox: bboxOf(V, [Infinity, Infinity, -Infinity, -Infinity]) };
  }
  const lines = {
    static: toF32(decodeArray(scene.field_lines?.static, 'field_lines.static').data),
    pickup: toF32(decodeArray(scene.field_lines?.pickup, 'field_lines.pickup').data),
  };
  // extent and view
  const e = scene.extent || {};
  const extent = {
    r_max: Number.isFinite(e.r_max) ? e.r_max : Number.isFinite(bb[2]) ? bb[2] : 1,
    z_min: Number.isFinite(e.z_min) ? e.z_min : Number.isFinite(bb[1]) ? bb[1] : -1,
    z_max: Number.isFinite(e.z_max) ? e.z_max : Number.isFinite(bb[3]) ? bb[3] : 1,
  };
  if (!(extent.r_max > 0)) extent.r_max = 1;
  if (!(extent.z_max > extent.z_min)) { extent.z_min -= 0.5; extent.z_max += 0.5; }
  const L = Math.max(extent.r_max, (extent.z_max - extent.z_min) / 2);
  const v = scene.view || {};
  const fb = field ? field.bbox : null;
  const view = {
    r_max: Number.isFinite(v.r_max) ? v.r_max : fb ? fb[2] : 1.3 * extent.r_max,
    z_min: Number.isFinite(v.z_min) ? v.z_min : fb ? fb[1] : extent.z_min - 0.3 * L,
    z_max: Number.isFinite(v.z_max) ? v.z_max : fb ? fb[3] : extent.z_max + 0.3 * L,
  };
  for (const R of regions) R.loops = processLoops(R.raw, L);
  const design = scene.design && typeof scene.design === 'object' ? scene.design : {};
  return {
    name: String(scene.name || design.name || 'design'),
    schema: scene.schema || '',
    description: typeof design.description === 'string' ? design.description : '',
    extent, view, L, regions, curves, field, lines,
    summary: scene.summary && typeof scene.summary === 'object' ? scene.summary : {},
    design, warnings,
  };
}

/* ------------------------------------------------------ point location grid */
class TriGrid {
  constructor(V, T) {
    this.V = V; this.T = T;
    const nt = T.length / 3;
    const b = bboxOf(V, [Infinity, Infinity, -Infinity, -Infinity]);
    const span = Math.max(b[2] - b[0], b[3] - b[1], 1e-30);
    this.x0 = b[0] - 1e-6 * span; this.y0 = b[1] - 1e-6 * span;
    const w = b[2] - b[0] + 2e-6 * span, hgt = b[3] - b[1] + 2e-6 * span;
    const n = clamp(Math.round(Math.sqrt(Math.max(nt, 1) / 2)), 1, 1024);
    this.nx = clamp(Math.round(n * Math.sqrt(w / hgt)), 1, 2048);
    this.ny = clamp(Math.round(n * Math.sqrt(hgt / w)), 1, 2048);
    this.dx = w / this.nx; this.dy = hgt / this.ny;
    const nc = this.nx * this.ny;
    const start = new Uint32Array(nc + 1);
    const cell = (t, fn) => {
      const a = T[3 * t], bb = T[3 * t + 1], c = T[3 * t + 2];
      const xs = [V[2 * a], V[2 * bb], V[2 * c]], ys = [V[2 * a + 1], V[2 * bb + 1], V[2 * c + 1]];
      const i0 = clamp(Math.floor((Math.min(...xs) - this.x0) / this.dx), 0, this.nx - 1);
      const i1 = clamp(Math.floor((Math.max(...xs) - this.x0) / this.dx), 0, this.nx - 1);
      const j0 = clamp(Math.floor((Math.min(...ys) - this.y0) / this.dy), 0, this.ny - 1);
      const j1 = clamp(Math.floor((Math.max(...ys) - this.y0) / this.dy), 0, this.ny - 1);
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) fn(j * this.nx + i);
    };
    for (let t = 0; t < nt; t++) cell(t, (c) => { start[c + 1]++; });
    for (let c = 0; c < nc; c++) start[c + 1] += start[c];
    const items = new Uint32Array(start[nc]);
    const cur = start.slice(0, nc);
    for (let t = 0; t < nt; t++) cell(t, (c) => { items[cur[c]++] = t; });
    this.start = start; this.items = items;
  }
  /** Triangle containing (x, y) and barycentric weights in `w`; -1 if none. */
  locate(x, y, w) {
    const i = Math.floor((x - this.x0) / this.dx), j = Math.floor((y - this.y0) / this.dy);
    if (i < 0 || j < 0 || i >= this.nx || j >= this.ny) return -1;
    const c = j * this.nx + i, V = this.V, T = this.T;
    let best = -1, bestMin = -1e-3;
    for (let k = this.start[c]; k < this.start[c + 1]; k++) {
      const t = this.items[k];
      const a = T[3 * t], b = T[3 * t + 1], cc = T[3 * t + 2];
      const xa = V[2 * a], ya = V[2 * a + 1], xb = V[2 * b], yb = V[2 * b + 1], xc = V[2 * cc], yc = V[2 * cc + 1];
      const d = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc);
      if (d === 0) continue;
      const l0 = ((yb - yc) * (x - xc) + (xc - xb) * (y - yc)) / d;
      const l1 = ((yc - ya) * (x - xc) + (xa - xc) * (y - yc)) / d;
      const l2 = 1 - l0 - l1;
      const mn = Math.min(l0, l1, l2);
      if (mn > bestMin) {
        bestMin = mn; best = t;
        if (w) { w[0] = l0; w[1] = l1; w[2] = l2; }
        if (mn >= 0) break;
      }
    }
    if (best >= 0 && w) { // clamp slightly-outside hits (curved boundaries) onto the triangle
      const s = Math.max(w[0], 0) + Math.max(w[1], 0) + Math.max(w[2], 0);
      for (let k = 0; k < 3; k++) w[k] = Math.max(w[k], 0) / s;
    }
    return best;
  }
}

/**
 * Point sampler for a prepared scene (see prepareScene): sample(r, z) returns
 * {r, z, region, inField, values}. `values` holds every field quantity,
 * interpolated linearly (barycentric weights) on the field triangulation;
 * `region` is the region object containing the point, or null (vacuum or
 * outside). Grids are built on first use.
 */
export function makeSampler(model) {
  let fieldGrid = null, regionGrid = null, owner = null;
  const bary = [0, 0, 0];
  const ensure = () => {
    if (regionGrid === null) {
      let nv = 0, nt = 0;
      for (const R of model.regions) { nv += R.V.length; nt += R.T.length; }
      const V = new Float32Array(nv), T = new Uint32Array(nt);
      owner = new Uint16Array(nt / 3);
      let ov = 0, ot = 0;
      model.regions.forEach((R, i) => {
        V.set(R.V, ov);
        for (let k = 0; k < R.T.length; k++) T[ot + k] = R.T[k] + ov / 2;
        owner.fill(i, ot / 3, (ot + R.T.length) / 3);
        ov += R.V.length; ot += R.T.length;
      });
      regionGrid = nt ? new TriGrid(V, T) : false;
    }
    if (fieldGrid === null) fieldGrid = model.field && model.field.T.length ? new TriGrid(model.field.V, model.field.T) : false;
  };
  const sample = (r, z) => {
    ensure();
    const out = { r, z, region: null, inField: false, values: {} };
    if (regionGrid) {
      const t = regionGrid.locate(r, z, null);
      if (t >= 0) out.region = model.regions[owner[t]];
    }
    if (fieldGrid) {
      const t = fieldGrid.locate(r, z, bary);
      if (t >= 0) {
        out.inField = true;
        const T = model.field.T, a = T[3 * t], b = T[3 * t + 1], c = T[3 * t + 2];
        for (const [q, arr] of Object.entries(model.field.q)) out.values[q] = bary[0] * arr[a] + bary[1] * arr[b] + bary[2] * arr[c];
      }
    }
    return out;
  };
  sample.fieldGrid = () => { ensure(); return fieldGrid || null; };
  return sample;
}

/* ------------------------------------------------------------ value mapping */
/**
 * Map values of quantity q to [0, 1]. Linear: [0, max] for magnitudes and
 * non-negative fluxes, symmetric [-M, M] for signed components. Log: log10 of
 * |value| between a floor and the maximum; the floor is max(max·10^-decades,
 * smallest positive value), rounded down to a decade.
 */
function makeMapping(stats, q, scale, decades, range) {
  const meta = qmeta(q);
  if (scale === 'log') {
    let hi = range ? Math.max(Math.abs(range[0]), Math.abs(range[1])) : stats.maxAbs;
    if (!(hi > 0)) hi = 1;
    let lo = range && range[0] > 0 ? range[0] : Math.max(hi * 10 ** -decades, stats.minPos || 0);
    lo = 10 ** Math.floor(Math.log10(lo) + 1e-9);
    if (!(lo < hi)) lo = hi / 10;
    const l0 = Math.log10(lo), dl = Math.log10(hi) - l0;
    return { log: true, abs: stats.min < 0, lo, hi, map: (v) => clamp((Math.log10(Math.max(Math.abs(v), lo)) - l0) / dl, 0, 1) };
  }
  let lo, hi;
  if (range) { [lo, hi] = range; }
  else if (stats.signed || (meta.kind === 'component' && stats.min < 0)) { hi = stats.maxAbs; lo = -hi; }
  else { lo = Math.min(0, stats.min); hi = Math.max(0, stats.max); }
  const uniform = !(hi > lo);
  if (uniform) { lo = lo - 0.5 || -0.5; hi = lo + 1; }
  const s = 1 / (hi - lo);
  return { log: false, abs: false, lo, hi, uniform, symmetric: lo === -hi, map: (v) => clamp((v - lo) * s, 0, 1) };
}

/* -------------------------------------------------------------- geometry */
function revolveGeometry(loops, phi0, dphi, segsPerTurn, creaseDeg = 35) {
  const m = Math.max(4, Math.ceil((Math.abs(dphi) / (2 * Math.PI)) * segsPerTurn));
  const cs = new Float64Array(m + 1), sn = new Float64Array(m + 1);
  for (let j = 0; j <= m; j++) { const p = phi0 + (dphi * j) / m; cs[j] = Math.cos(p); sn[j] = Math.sin(p); }
  const creaseCos = Math.cos(creaseDeg * DEG);
  let nEdges = 0;
  for (const lp of loops) nEdges += lp.pts.length >> 1;
  const nv = nEdges * 2 * (m + 1);
  const pos = new Float32Array(3 * nv), nor = new Float32Array(3 * nv);
  const idx = nv > 65535 ? new Uint32Array(nEdges * m * 6) : new Uint16Array(nEdges * m * 6);
  let v = 0, f = 0;
  for (const lp of loops) {
    const P = lp.pts, n = P.length >> 1;
    const en = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const dx = P[2 * j] - P[2 * i], dz = P[2 * j + 1] - P[2 * i + 1];
      const l = Math.hypot(dx, dz) || 1;
      en[2 * i] = dz / l; en[2 * i + 1] = -dx / l;
    }
    const vn = new Float64Array(2 * n), hard = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const p = (i - 1 + n) % n;
      const ax = en[2 * p], az = en[2 * p + 1], bx = en[2 * i], bz = en[2 * i + 1];
      if (ax * bx + az * bz < creaseCos) hard[i] = 1;
      const sx = ax + bx, sz = az + bz, l = Math.hypot(sx, sz) || 1;
      vn[2 * i] = sx / l; vn[2 * i + 1] = sz / l;
    }
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const r0 = P[2 * i], z0 = P[2 * i + 1], r1 = P[2 * j], z1 = P[2 * j + 1];
      const n0r = hard[i] ? en[2 * i] : vn[2 * i], n0z = hard[i] ? en[2 * i + 1] : vn[2 * i + 1];
      const n1r = hard[j] ? en[2 * i] : vn[2 * j], n1z = hard[j] ? en[2 * i + 1] : vn[2 * j + 1];
      const base = v;
      for (let k = 0; k <= m; k++) {
        let o = 3 * v;
        pos[o] = r0 * cs[k]; pos[o + 1] = r0 * sn[k]; pos[o + 2] = z0;
        nor[o] = n0r * cs[k]; nor[o + 1] = n0r * sn[k]; nor[o + 2] = n0z;
        v++; o = 3 * v;
        pos[o] = r1 * cs[k]; pos[o + 1] = r1 * sn[k]; pos[o + 2] = z1;
        nor[o] = n1r * cs[k]; nor[o + 1] = n1r * sn[k]; nor[o + 2] = n1z;
        v++;
      }
      for (let k = 0; k < m; k++) {
        const a = base + 2 * k, b = a + 1, a2 = a + 2, b2 = a + 3;
        if (dphi >= 0) { idx[f++] = a; idx[f++] = a2; idx[f++] = b; idx[f++] = b; idx[f++] = a2; idx[f++] = b2; }
        else { idx[f++] = a; idx[f++] = b; idx[f++] = a2; idx[f++] = b; idx[f++] = b2; idx[f++] = a2; }
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeBoundingSphere();
  return g;
}
function revolveSegments(S, phi0, dphi, segsPerTurn) {
  const loops = [];
  for (let i = 0; i + 3 < S.length; i += 4) loops.push({ pts: Float64Array.of(S[i], S[i + 1], S[i + 2], S[i + 3]) });
  // a two-point "loop" revolves its segment twice (there and back); harmless for a double-sided sheet
  return revolveGeometry(loops, phi0, dphi, segsPerTurn, 90);
}
function planeGeometry(V, T, withUV) {
  const n = V.length >> 1;
  const pos = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { pos[3 * i] = V[2 * i]; pos[3 * i + 1] = V[2 * i + 1]; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  if (withUV) g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(2 * n), 2));
  g.setIndex(new THREE.BufferAttribute(T, 1));
  g.computeBoundingSphere();
  return g;
}
function segmentsGeometry(S) {
  const n = S.length >> 1;
  const pos = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { pos[3 * i] = S[2 * i]; pos[3 * i + 1] = S[2 * i + 1]; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  return g;
}
function loopsSegments(loops) {
  const out = [];
  for (const lp of loops) {
    const P = lp.pts, n = P.length >> 1;
    for (let i = 0; i < n; i++) { const j = (i + 1) % n; out.push(P[2 * i], P[2 * i + 1], P[2 * j], P[2 * j + 1]); }
  }
  return Float32Array.from(out);
}
function planeBasis(phi) {
  const c = Math.cos(phi), s = Math.sin(phi);
  return new THREE.Matrix4().makeBasis(new THREE.Vector3(c, s, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(s, -c, 0));
}
function lutTextureData(name) {
  const b = lutBytes(name), d = new Uint8Array(1024);
  for (let i = 0; i < 256; i++) { d[4 * i] = b[3 * i]; d[4 * i + 1] = b[3 * i + 1]; d[4 * i + 2] = b[3 * i + 2]; d[4 * i + 3] = 255; }
  return d;
}

/* ------------------------------------------------------------------- CSS */
const CSS = `
.axf-root{position:relative;width:100%;height:100%;min-height:260px;overflow:hidden;font:13px/1.45 ${FONT};color:var(--fg);background:var(--bgimg);-webkit-font-smoothing:antialiased;-webkit-tap-highlight-color:transparent;font-variant-numeric:tabular-nums}
.axf-root *{box-sizing:border-box}
.axf-root[data-theme=dark]{--bgimg:radial-gradient(130% 95% at 50% 32%,#1b2332 0%,#0d1117 72%);--fg:#e6edf3;--fg2:#9da7b3;--fg3:#6e7781;--panel:rgba(22,27,34,.80);--solid:#161b22;--border:rgba(240,246,252,.11);--hover:rgba(240,246,252,.07);--accent:#58a6ff;--accentfg:#0d1117;--good:#3fb950;--warn:#d29922;--shadow:0 10px 28px rgba(0,0,0,.38);color-scheme:dark}
.axf-root[data-theme=light]{--bgimg:radial-gradient(130% 95% at 50% 32%,#ffffff 0%,#e8ecf1 75%);--fg:#1f2328;--fg2:#57606a;--fg3:#8c959f;--panel:rgba(255,255,255,.84);--solid:#ffffff;--border:rgba(31,35,40,.13);--hover:rgba(31,35,40,.05);--accent:#0969da;--accentfg:#ffffff;--good:#1a7f37;--warn:#9a6700;--shadow:0 10px 28px rgba(110,119,129,.22);color-scheme:light}
.axf-canvas,.axf-overlay{position:absolute;left:0;top:0;width:100%;height:100%;display:block}
.axf-canvas{outline:none;touch-action:none}
.axf-overlay{pointer-events:none}
.axf-glass{background:var(--panel);border:1px solid var(--border);-webkit-backdrop-filter:blur(12px);backdrop-filter:blur(12px);box-shadow:var(--shadow)}
.axf-header{position:absolute;left:0;right:0;top:0;display:flex;align-items:flex-start;justify-content:space-between;gap:10px;padding:12px 12px 0;pointer-events:none;z-index:3}
.axf-header>*{pointer-events:auto}
.axf-title{min-width:0;max-width:min(560px,52%);padding:7px 12px 8px;border-radius:12px}
.axf-name{font-size:16px;font-weight:650;letter-spacing:.005em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.axf-name small{font-size:11px;font-weight:550;color:var(--fg3);margin-left:8px;letter-spacing:.04em;text-transform:uppercase}
.axf-sub{font-size:12px;color:var(--fg2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.axf-tools{display:flex;gap:6px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
.axf-seg{display:inline-flex;border-radius:9px;overflow:hidden;padding:2px;gap:2px}
.axf-seg button{all:unset;cursor:pointer;padding:4px 11px;border-radius:7px;font-size:12px;font-weight:600;color:var(--fg2);white-space:nowrap}
.axf-seg button:hover{color:var(--fg);background:var(--hover)}
.axf-seg button[aria-pressed=true]{background:var(--accent);color:var(--accentfg)}
.axf-seg button:disabled{opacity:.38;cursor:not-allowed}
.axf-seg button:focus-visible,.axf-btn:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.axf-btn{all:unset;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;gap:5px;height:30px;min-width:30px;padding:0 9px;border-radius:9px;color:var(--fg);font-size:12px;font-weight:600;white-space:nowrap}
.axf-btn:hover{background:var(--hover)}
.axf-btn svg{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.axf-btn[aria-pressed=true]{color:var(--accent)}
.axf-badge{font-size:11px;font-weight:650;padding:3px 8px;border-radius:999px;letter-spacing:.03em}
.axf-badge[data-kind=ok]{color:var(--good)} .axf-badge[data-kind=warn]{color:var(--warn)} .axf-badge[data-kind=busy]{color:var(--accent)}
.axf-panel{position:absolute;top:70px;max-height:calc(100% - 70px - 104px);overflow:auto;overscroll-behavior:contain;padding:12px 12px 10px;border-radius:14px;z-index:2;scrollbar-width:thin}
.axf-panel[hidden]{display:none}
.axf-controls{left:12px;width:250px}
.axf-results{right:12px;width:340px}
.axf-ph{display:flex;align-items:center;justify-content:space-between;font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--fg3);margin:0 0 8px}
.axf-ph button{all:unset;cursor:pointer;color:var(--fg3);font-size:16px;line-height:1;padding:0 2px}
.axf-field{margin:0 0 10px}
.axf-lbl{display:flex;justify-content:space-between;font-size:12px;font-weight:600;color:var(--fg2);margin-bottom:4px}
.axf-lbl span{color:var(--fg);font-weight:600}
.axf-select{width:100%;font:inherit;font-size:12.5px;color:var(--fg);background:var(--solid);border:1px solid var(--border);border-radius:8px;padding:5px 8px}
.axf-hint{font-size:11.5px;color:var(--fg3);margin-top:4px;line-height:1.35}
.axf-row{display:flex;align-items:center;justify-content:space-between;gap:8px;margin:0 0 10px}
.axf-row .axf-lbl{margin:0}
.axf-row .axf-seg{border:1px solid var(--border);background:var(--solid)}
.axf-range{width:100%;accent-color:var(--accent)}
.axf-check{display:flex;align-items:center;gap:8px;font-size:12.5px;padding:3px 2px;border-radius:6px;cursor:pointer}
.axf-check:hover{background:var(--hover)}
.axf-check input{accent-color:var(--accent);margin:0}
.axf-check.dis{opacity:.45;cursor:default}
.axf-check small{color:var(--fg3);margin-left:auto;font-size:11px;white-space:nowrap}
.axf-sw{display:inline-block;width:12px;height:12px;border-radius:3px;flex:none;border:1px solid rgba(127,127,127,.35)}
.axf-sw.line{width:14px;height:10px;border:none;border-radius:3px;background-color:#29303a;background-repeat:no-repeat;background-position:center;background-size:100% 2px}
.axf-cal{font-family:"Times New Roman",Times,"Nimbus Roman",Georgia,serif;font-style:italic;font-weight:600;font-size:1.08em}
.axf-sep{height:1px;background:var(--border);margin:10px 0}
.axf-res details{border-top:1px solid var(--border);padding:7px 0 5px}
.axf-res details:first-of-type{border-top:none;padding-top:0}
.axf-res summary{cursor:pointer;font-size:12px;font-weight:700;color:var(--fg);list-style:none;display:flex;align-items:center;gap:6px;margin-bottom:4px}
.axf-res summary::-webkit-details-marker{display:none}
.axf-res summary::before{content:'';width:6px;height:6px;border-right:1.6px solid var(--fg3);border-bottom:1.6px solid var(--fg3);transform:rotate(-45deg);transition:transform .15s;margin:0 3px 0 1px}
.axf-res details[open]>summary::before{transform:rotate(45deg)}
.axf-kv{display:grid;grid-template-columns:minmax(0,1fr) auto;column-gap:10px;row-gap:2px;font-size:12px}
.axf-kv .k{color:var(--fg2);min-width:0}
.axf-kv .v{text-align:right;color:var(--fg);font-weight:550;white-space:nowrap}
.axf-kv .s{grid-column:1/-1;text-align:right;color:var(--fg3);font-size:11px;margin-top:-2px}
.axf-kv .hd{grid-column:1/-1;font-size:11px;font-weight:700;color:var(--fg3);letter-spacing:.04em;text-transform:uppercase;margin-top:6px}
.axf-bench{border:1px solid var(--border);border-radius:10px;padding:7px 9px;margin:6px 0}
.axf-bench h4{margin:0 0 3px;font-size:12.5px;display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.axf-bench h4 small{font-weight:500;color:var(--fg3);font-size:11px}
.axf-pill{display:inline-block;padding:0 6px;border-radius:999px;font-size:11px;font-weight:650}
.axf-pill.ok{color:var(--good);background:color-mix(in srgb,var(--good) 14%,transparent)}
.axf-pill.short{color:var(--warn);background:color-mix(in srgb,var(--warn) 14%,transparent)}
.axf-note{font-size:11.5px;color:var(--fg3);line-height:1.4;margin:4px 0}
.axf-res pre{font-size:10.5px;max-height:220px;overflow:auto;background:var(--solid);border:1px solid var(--border);border-radius:8px;padding:6px;user-select:text}
.axf-legend{position:absolute;left:12px;bottom:12px;max-width:min(380px,46%);padding:8px 11px;border-radius:12px;font-size:11.5px;z-index:1}
.axf-legend .it{display:inline-flex;align-items:center;gap:5px;margin:1px 10px 1px 0;white-space:nowrap;color:var(--fg2)}
.axf-legend .ln{margin-top:4px;color:var(--fg3);line-height:1.35}
.axf-cbar{position:absolute;right:12px;bottom:12px;width:320px;height:74px;border-radius:12px;z-index:1;overflow:hidden}
.axf-cbar canvas{display:block;width:100%;height:100%}
.axf-probe{position:absolute;left:0;top:0;pointer-events:none;padding:8px 10px;border-radius:10px;font-size:12px;line-height:1.45;max-width:320px;z-index:6;background:var(--solid);border:1px solid var(--border);box-shadow:var(--shadow)}
.axf-probe[hidden]{display:none}
.axf-probe .c{color:var(--fg2)}
.axf-probe .r{display:flex;align-items:center;gap:6px;font-weight:600;margin:2px 0}
.axf-probe .m{font-size:13.5px;font-weight:700}
.axf-probe .o{color:var(--fg2);font-size:11.5px}
.axf-probe .n{color:var(--fg3);font-size:11.5px}
.axf-msg{cursor:pointer;position:absolute;left:50%;top:70px;transform:translateX(-50%);max-width:min(560px,90%);padding:7px 12px;border-radius:10px;font-size:12px;color:var(--warn);z-index:4}
.axf-msg[hidden]{display:none}
.axf-root.narrow .axf-title{max-width:100%;flex:1 1 auto;padding:6px 10px}
.axf-root.narrow .axf-sub{display:none}
.axf-root.narrow .axf-header{flex-wrap:wrap;padding:8px 8px 0}
.axf-root.narrow .axf-panel{left:8px;right:8px;width:auto;top:auto;bottom:92px;max-height:min(60%,calc(100% - 180px))}
.axf-root.narrow .axf-legend{display:none}
.axf-root.narrow .axf-cbar{left:8px;right:8px;bottom:8px;width:auto}
.axf-root.narrow .axf-msg{top:100px}
.axf-root.noui .axf-header,.axf-root.noui .axf-panel,.axf-root.noui .axf-legend{display:none}
`;
function injectCSS(doc) {
  if (doc.getElementById('axf-viewer-css')) return;
  const st = doc.createElement('style');
  st.id = 'axf-viewer-css';
  st.textContent = CSS;
  doc.head.appendChild(st);
}
const ICONS = {
  reset: '<svg viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg>',
  camera: '<svg viewBox="0 0 24 24"><path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/></svg>',
  sliders: '<svg viewBox="0 0 24 24"><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/></svg>',
  results: '<svg viewBox="0 0 24 24"><path d="M5 4h14v16H5z"/><path d="M8 8h8M8 12h8M8 16h5"/></svg>',
};

/* ---------------------------------------------------------------- options */
const DEFAULTS = {
  mode: '3d', quantity: null, scale: 'linear', colormap: 'auto', cutaway: 90, azimuthStart: 0,
  transparent: false, fieldBeyondDevice: false, theme: 'auto', ui: true, panels: true, results: true,
  hide: [], show: [], logDecades: 4, range: null, renderer: 'auto', preserveDrawingBuffer: true,
  maxPixelRatio: 2, segments: 160, title: null, elevation: 24, probeAt: null,
};
function normalizeOptions(o) {
  const opts = Object.assign({}, DEFAULTS, o || {});
  if (o && Number.isFinite(o.azimuth) && !Number.isFinite(o.cutaway)) opts.cutaway = 360 - o.azimuth;
  opts.cutaway = clamp(Number(opts.cutaway) || 0, 0, 359);
  opts.mode = opts.mode === 'section' || opts.mode === '2d' ? 'section' : '3d';
  opts.scale = opts.scale === 'log' ? 'log' : 'linear';
  opts.logDecades = clamp(Number(opts.logDecades) || 4, 1, 12);
  if (typeof opts.hide === 'string') opts.hide = opts.hide.split(',').filter(Boolean);
  if (typeof opts.show === 'string') opts.show = opts.show.split(',').filter(Boolean);
  return opts;
}

/**
 * Viewer options from URL query parameters (used by the HTML shell):
 * mode=3d|section, q=<quantity>, scale=linear|log, cmap=<colormap>,
 * cut=<open wedge, deg>, az=<revolved range, deg>, theme=auto|dark|light,
 * ui=0, panels=0, results=0, hide=a,b, show=a,b, transparent=1, beyond=1,
 * decades=<n>, renderer=webgl|canvas, elev=<deg>, probe=<x>,<y> (pinned probe at
 * pixel x, y), shot=1.
 */
export function optionsFromURL(search) {
  const s = search ?? (typeof location !== 'undefined' ? location.search : '');
  const p = new URLSearchParams(s);
  const o = {};
  const flag = (k) => p.has(k) && !/^(0|false|no|off)$/i.test(p.get(k));
  if (p.has('mode')) o.mode = p.get('mode');
  if (p.has('q') || p.has('quantity')) o.quantity = p.get('q') || p.get('quantity');
  if (p.has('scale')) o.scale = p.get('scale');
  if (p.has('cmap') || p.has('colormap')) o.colormap = p.get('cmap') || p.get('colormap');
  if (p.has('cut') || p.has('cutaway')) o.cutaway = parseFloat(p.get('cut') ?? p.get('cutaway'));
  if (p.has('az') || p.has('azimuth')) o.azimuth = parseFloat(p.get('az') ?? p.get('azimuth'));
  if (p.has('theme')) o.theme = p.get('theme');
  if (p.has('ui')) o.ui = flag('ui');
  if (p.has('panels')) o.panels = flag('panels');
  if (p.has('results')) o.results = flag('results');
  if (p.has('hide')) o.hide = p.get('hide').split(',').filter(Boolean);
  if (p.has('show')) o.show = p.get('show').split(',').filter(Boolean);
  if (p.has('transparent')) o.transparent = flag('transparent');
  if (p.has('beyond')) o.fieldBeyondDevice = flag('beyond');
  if (p.has('decades')) o.logDecades = parseFloat(p.get('decades'));
  if (p.has('renderer')) o.renderer = p.get('renderer');
  if (p.has('elev')) o.elevation = parseFloat(p.get('elev'));
  if (p.has('probe')) o.probeAt = p.get('probe').split(',').map(parseFloat);
  if (flag('shot')) o.preserveDrawingBuffer = true;
  return o;
}

/* ============================================================ createViewer */
/**
 * Create a viewer inside `container` for a parsed axionfem.scene/1 object.
 * Returns an API object: setQuantity, setScale, setMode, setCutaway,
 * toggleLayer, screenshot, dispose, plus setColormap, setScene, setRange,
 * setTransparent, setTheme, setStatus, resetView, getState, layers, probe.
 */
export function createViewer(container, scene, options = {}) {
  if (!container || typeof container.appendChild !== 'function') throw new Error('createViewer: container must be a DOM element');
  if (scene && scene.symmetry === '3d') return createViewer3D(container, scene, options);
  const doc = container.ownerDocument || document;
  const win = doc.defaultView || window;
  const opts = normalizeOptions(options);
  injectCSS(doc);
  let model = prepareScene(scene);

  const state = {
    mode: opts.mode, quantity: null, scale: opts.scale, colormap: opts.colormap, cutaway: opts.cutaway,
    phi0: opts.azimuthStart * DEG, transparent: !!opts.transparent, beyond: !!opts.fieldBeyondDevice,
    theme: opts.theme, hidden: new Set(), linesAuto: true, range: opts.range, decades: opts.logDecades,
    panels: { controls: !!opts.panels, results: !!opts.panels && !!opts.results }, status: null,
  };
  const listeners = [];
  const on = (target, ev, fn, o) => { target.addEventListener(ev, fn, o); listeners.push(() => target.removeEventListener(ev, fn, o)); };

  /* ---------------------------------------------------------------- DOM */
  const root = h('div', { class: 'axf-root' + (opts.ui ? '' : ' noui'), tabindex: '-1' });
  container.appendChild(root);
  let dpr = Math.min(win.devicePixelRatio || 1, opts.maxPixelRatio);

  // renderer (WebGL, with a Canvas2D section fallback)
  let renderer = null, glError = null;
  if (opts.renderer !== 'canvas') {
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: !!opts.preserveDrawingBuffer, powerPreference: 'high-performance' });
    } catch (e) { glError = e; renderer = null; }
  }
  const webgl = !!renderer;
  if (!webgl) state.mode = 'section';
  const canvas = webgl ? renderer.domElement : h('canvas');
  canvas.classList.add('axf-canvas');
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', `axionFEM view of ${model.name}`);
  root.appendChild(canvas);
  const overlay = h('canvas', { class: 'axf-overlay' });
  root.appendChild(overlay);
  const octx = overlay.getContext('2d');
  const c2d = webgl ? null : canvas.getContext('2d');

  /* --------------------------------------------------------- three.js setup */
  let W = 1, H = 1;
  const scene3 = new THREE.Scene(), scene2 = new THREE.Scene();
  const cam3 = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
  cam3.up.set(0, 0, 1);
  const cam2 = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.001, 1000);
  scene3.add(cam3);
  let controls3 = null, controls2 = null, envTex = null;
  const lutData = lutTextureData('viridis');
  const lutTex = new THREE.DataTexture(lutData, 256, 1, THREE.RGBAFormat);
  lutTex.colorSpace = THREE.SRGBColorSpace;
  lutTex.magFilter = THREE.LinearFilter; lutTex.minFilter = THREE.LinearFilter;
  lutTex.wrapS = lutTex.wrapT = THREE.ClampToEdgeWrapping;
  lutTex.generateMipmaps = false;
  lutTex.needsUpdate = true;
  const axisMat = new THREE.LineBasicMaterial({ color: THEMES.dark.axis, transparent: true, opacity: 0.9 });
  const axisMat2 = new THREE.LineDashedMaterial({ color: THEMES.dark.axis, transparent: true, opacity: 0.55, dashSize: 1, gapSize: 1, depthTest: false });
  const coneMat = new THREE.MeshBasicMaterial({ color: THEMES.dark.axis });
  if (webgl) {
    renderer.setPixelRatio(dpr);
    renderer.localClippingEnabled = true;
    renderer.toneMapping = THREE.NeutralToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.setClearColor(0x000000, 0);
    try {
      const pm = new THREE.PMREMGenerator(renderer);
      const room = new RoomEnvironment();
      envTex = pm.fromScene(room, 0.04).texture;
      room.traverse((o) => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
      pm.dispose();
      scene3.environment = envTex;
      scene3.environmentIntensity = 0.85;
    } catch (e) { /* environment lighting is optional */ }
    const hemi = new THREE.HemisphereLight(0xe8eef7, 0x3a3f47, 0.55);
    hemi.position.set(0, 0, 1);
    scene3.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(-0.55, 0.75, 1.0);
    cam3.add(key); cam3.add(key.target);
    key.target.position.set(0, 0, -1);
    const rim = new THREE.DirectionalLight(0xc9d8ff, 0.45);
    rim.position.set(0.3, 0.5, -1.0);
    scene3.add(rim);
    controls3 = new OrbitControls(cam3, canvas);
    controls3.enableDamping = true; controls3.dampingFactor = 0.12;
    controls3.rotateSpeed = 0.8; controls3.zoomToCursor = true;
    controls2 = new OrbitControls(cam2, canvas);
    controls2.enableRotate = false; controls2.enableDamping = false; controls2.zoomToCursor = true;
    controls2.screenSpacePanning = true;
    controls2.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    controls2.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_PAN };
    controls3.enabled = state.mode === '3d';
    controls2.enabled = state.mode === 'section';
  }
  const userMoved = { '3d': false, section: false };
  if (webgl) {
    const startPose = { '3d': null, section: null };
    const pose = (cam) => [cam.position.x, cam.position.y, cam.position.z, cam.zoom].join(',');
    on(controls3, 'start', () => { startPose['3d'] = pose(cam3); hideProbe(); });
    on(controls2, 'start', () => { startPose.section = pose(cam2); hideProbe(); });
    on(controls3, 'end', () => { if (pose(cam3) !== startPose['3d']) { userMoved['3d'] = true; unpin(); } });
    on(controls2, 'end', () => { if (pose(cam2) !== startPose.section) { userMoved.section = true; unpin(); } });
    on(controls3, 'change', () => { dirty = true; });
    on(controls2, 'change', () => { dirty = true; });
  }

  /* ------------------------------------------------------- scene content */
  let C = null; // current content (rebuilt by setScene)
  function buildContent() {
    const m = model, L = m.L;
    const c = { disposables: [], surf: [], planes: [], regionFace: new Map(), group3: new THREE.Group(), group2: new THREE.Group(), fieldMats3: [], lineMats3: [], field2: [], lines2: { static: [], pickup: [] }, regions2: [], outlines3: [], outlines2: [], curves3: [], curves2: [], lines3: { static: [], pickup: [] }, faces3: [], field3: [] };
    const track = (o) => { c.disposables.push(o); return o; };
    scene3.add(c.group3); scene2.add(c.group2);
    // shared geometries
    c.fieldGeom = m.field ? track(planeGeometry(m.field.V, m.field.T, true)) : null;
    c.regionGeom = m.regions.map((R) => track(planeGeometry(R.V, R.T, false)));
    c.outlineGeom = m.regions.map((R) => track(segmentsGeometry(loopsSegments(R.loops))));
    c.lineGeom = { static: m.lines.static.length ? track(segmentsGeometry(m.lines.static)) : null, pickup: m.lines.pickup.length ? track(segmentsGeometry(m.lines.pickup)) : null };
    c.curveGeom = m.curves.map((cv) => track(segmentsGeometry(cv.S)));
    // ---- 3D: revolved surfaces
    m.regions.forEach((R) => {
      const st = roleStyle(R.role);
      const col = R.role === 'superconductor' ? st.color : R.color;
      const mat = track(new THREE.MeshStandardMaterial({ color: col, metalness: st.metal, roughness: st.rough, side: THREE.DoubleSide }));
      mat.userData.baseOpacity = st.opacity ?? 1;
      const mesh = new THREE.Mesh(new THREE.BufferGeometry(), mat);
      mesh.userData = { kind: 'surface', region: R };
      c.group3.add(mesh); c.surf.push(mesh);
    });
    m.curves.forEach((cv) => {
      const mat = track(new THREE.MeshStandardMaterial({ color: ROLE.superconductor.color, metalness: 0.9, roughness: 0.3, side: THREE.DoubleSide }));
      const mesh = new THREE.Mesh(new THREE.BufferGeometry(), mat);
      mesh.userData = { kind: 'curve', curve: cv };
      c.group3.add(mesh); c.curves3.push(mesh);
    });
    // ---- 3D: the two cut half-planes
    const dFace = 7e-4 * L, dOut = 1.4e-3 * L, dLine = 2.1e-3 * L;
    for (let k = 0; k < 2; k++) {
      const grp = new THREE.Group();
      grp.matrixAutoUpdate = false;
      const sgn = k === 0 ? 1 : -1;
      c.group3.add(grp); c.planes.push(grp);
      if (c.fieldGeom) {
        const mat = track(new THREE.MeshBasicMaterial({ map: lutTex, side: THREE.DoubleSide, toneMapped: false, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 2 }));
        const mesh = new THREE.Mesh(c.fieldGeom, mat);
        mesh.userData.kind = 'field';
        grp.add(mesh); c.field3.push(mesh); c.fieldMats3.push(mat);
      }
      m.regions.forEach((R, i) => {
        const st = roleStyle(R.role);
        const col = R.role === 'superconductor' ? st.color : R.color;
        const mat = track(new THREE.MeshStandardMaterial({ color: col, metalness: st.metal * 0.55, roughness: Math.min(1, st.rough + 0.25), side: THREE.DoubleSide, transparent: (st.opacity ?? 1) < 1, opacity: st.opacity ?? 1 }));
        const mesh = new THREE.Mesh(c.regionGeom[i], mat);
        mesh.position.z = sgn * dFace;
        mesh.userData = { kind: 'face', region: R };
        grp.add(mesh); c.faces3.push(mesh);
        const omat = track(new THREE.LineBasicMaterial({ color: THEMES.dark.outline, transparent: true, opacity: 0.55 }));
        const ol = new THREE.LineSegments(c.outlineGeom[i], omat);
        ol.position.z = sgn * dOut;
        ol.userData = { region: R };
        grp.add(ol); c.outlines3.push(ol);
      });
      for (const fam of ['static', 'pickup']) {
        if (!c.lineGeom[fam]) continue;
        const mat = track(new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.75, toneMapped: false }));
        const ls = new THREE.LineSegments(c.lineGeom[fam], mat);
        ls.position.z = sgn * dLine;
        grp.add(ls); c.lines3[fam].push(ls); c.lineMats3.push(mat);
      }
      m.curves.forEach((cv, i) => {
        const mat = track(new THREE.LineBasicMaterial({ color: '#e9eef5' }));
        const ls = new THREE.LineSegments(c.curveGeom[i], mat);
        ls.position.z = sgn * dOut;
        ls.userData = { curve: cv };
        grp.add(ls); c.curves3.push(ls);
      });
    }
    // ---- 3D: axis, ticks and radial ruler
    c.axes3 = new THREE.Group();
    c.group3.add(c.axes3);
    c.axisLabels = [];
    buildAxes3(c, track);
    // ---- section (2D) content, clipped to the view window
    const vw = m.view;
    const clip2 = [
      new THREE.Plane(new THREE.Vector3(-1, 0, 0), vw.r_max), new THREE.Plane(new THREE.Vector3(1, 0, 0), vw.r_max),
      new THREE.Plane(new THREE.Vector3(0, 1, 0), -vw.z_min), new THREE.Plane(new THREE.Vector3(0, -1, 0), vw.z_max),
    ];
    for (const side of [1, -1]) {
      const g = new THREE.Group();
      g.scale.x = side;
      c.group2.add(g);
      if (c.fieldGeom) {
        const mat = track(new THREE.MeshBasicMaterial({ map: lutTex, side: THREE.DoubleSide, toneMapped: false, depthTest: false, depthWrite: false }));
        mat.clippingPlanes = clip2;
        const mesh = new THREE.Mesh(c.fieldGeom, mat);
        mesh.renderOrder = 0;
        g.add(mesh); c.field2.push(mesh);
      }
      m.regions.forEach((R, i) => {
        const st = roleStyle(R.role);
        const col = R.role === 'superconductor' ? st.flat : R.color;
        const mat = track(new THREE.MeshBasicMaterial({ color: col, side: THREE.DoubleSide, transparent: st.alpha2d < 1, opacity: st.alpha2d, depthTest: false, depthWrite: false }));
        const mesh = new THREE.Mesh(c.regionGeom[i], mat);
        mesh.renderOrder = 1;
        mesh.userData = { region: R };
        g.add(mesh); c.regions2.push(mesh);
        const omat = track(new THREE.LineBasicMaterial({ color: R.role === 'superconductor' ? '#dfe5ec' : mixHex(R.color, '#ffffff', 0.25), transparent: true, opacity: 0.95, depthTest: false }));
        const ol = new THREE.LineSegments(c.outlineGeom[i], omat);
        ol.renderOrder = 2;
        ol.userData = { region: R };
        g.add(ol); c.outlines2.push(ol);
      });
      for (const fam of ['static', 'pickup']) {
        if (!c.lineGeom[fam]) continue;
        const mat = track(new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.75, depthTest: false, toneMapped: false }));
        mat.clippingPlanes = clip2;
        const ls = new THREE.LineSegments(c.lineGeom[fam], mat);
        ls.renderOrder = 3;
        g.add(ls); c.lines2[fam].push(ls);
      }
      m.curves.forEach((cv, i) => {
        const mat = track(new THREE.LineBasicMaterial({ color: '#e9eef5', depthTest: false }));
        const ls = new THREE.LineSegments(c.curveGeom[i], mat);
        ls.renderOrder = 4; ls.userData = { curve: cv };
        g.add(ls); c.curves2.push(ls);
      });
    }
    const v = m.view;
    const ag = track(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, v.z_min, 0), new THREE.Vector3(0, v.z_max, 0)]));
    c.axis2 = new THREE.Line(ag, axisMat2);
    c.axis2.computeLineDistances();
    c.axis2.renderOrder = 5;
    axisMat2.dashSize = 0.02 * m.L; axisMat2.gapSize = 0.014 * m.L;
    c.group2.add(c.axis2);
    // probe grids (lazy)
    c.sampler = makeSampler(m);
    c.lineSamples = { static: null, pickup: null };
    c.lineColors = { static: null, pickup: null };
    return c;
  }
  function buildAxes3(c, track) {
    const m = model, L = m.L, e = m.extent;
    const zlo = e.z_min - 0.16 * L, zhi = e.z_max + 0.3 * L;
    const pts = [new THREE.Vector3(0, 0, zlo), new THREE.Vector3(0, 0, zhi)];
    c.axes3.add(new THREE.Line(track(new THREE.BufferGeometry().setFromPoints(pts)), axisMat));
    const cone = new THREE.Mesh(track(new THREE.ConeGeometry(0.018 * L, 0.07 * L, 20)), coneMat);
    cone.rotation.x = Math.PI / 2;
    cone.position.set(0, 0, zhi + 0.035 * L);
    c.axes3.add(cone);
    c.axisLabels.push({ p: new THREE.Vector3(0, 0, zhi + 0.1 * L), text: 'z', bold: true });
    // z ticks: short dashes along the open-wedge bisector (set in layoutAxes3)
    const zt = niceTicks(e.z_min, e.z_max, 4);
    c.zTicks = zt.ticks.map((t, i) => ({ z: t, text: tickText(t, zt.step) + (i === zt.ticks.length - 1 ? ' m' : '') }));
    const rt = niceTicks(0, e.r_max, 4);
    c.rTicks = rt.ticks.map((t, i) => ({ r: t, text: tickText(t, rt.step) + (i === rt.ticks.length - 1 ? ' m' : '') }));
    c.rRulerEnd = rt.ticks[rt.ticks.length - 1];
    c.tickGeom = track(new THREE.BufferGeometry());
    c.tickLines = new THREE.LineSegments(c.tickGeom, axisMat);
    c.axes3.add(c.tickLines);
  }
  function layoutAxes3() {
    // place z-axis ticks and the radial ruler relative to the current cut planes
    const c = C, m = model, L = m.L, e = m.extent;
    const cut = state.cutaway * DEG, phi0 = state.phi0;
    const bis = phi0 - cut / 2; // open wedge bisector
    const bx = Math.cos(bis), by = Math.sin(bis);
    const u0x = Math.cos(phi0), u0y = Math.sin(phi0);
    const n0x = Math.sin(phi0), n0y = -Math.cos(phi0); // open side of the first plane
    const segs = [];
    const tl = 0.025 * L;
    c.labels3 = c.axisLabels.slice();
    for (const t of c.zTicks) {
      segs.push(0, 0, t.z, bx * tl, by * tl, t.z);
      c.labels3.push({ p: new THREE.Vector3(bx * tl * 2.2, by * tl * 2.2, t.z), text: t.text, align: 'left' });
    }
    const zr = e.z_min - 0.07 * L, off = 0.004 * L;
    const ox = n0x * off, oy = n0y * off;
    segs.push(ox, oy, zr, u0x * c.rRulerEnd + ox, u0y * c.rRulerEnd + oy, zr);
    for (const t of c.rTicks) {
      const x = u0x * t.r + ox, y = u0y * t.r + oy;
      segs.push(x, y, zr, x, y, zr - 0.022 * L);
      c.labels3.push({ p: new THREE.Vector3(x, y, zr - 0.06 * L), text: t.text, align: 'center' });
    }
    c.tickGeom.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(segs), 3));
    c.tickGeom.computeBoundingSphere();
    c.axes3.children[0].position.set(bx * 0.002 * L, by * 0.002 * L, 0);
  }
  function disposeContent(c) {
    scene3.remove(c.group3); scene2.remove(c.group2);
    for (const s of c.surf) s.geometry.dispose();
    for (const s of c.curves3) if (s.userData.kind === 'curve') s.geometry.dispose();
    for (const d of c.disposables) d.dispose?.();
  }

  /* --------------------------------------------------- derived updates */
  function rebuildSurfaces() {
    const dphi = (360 - state.cutaway) * DEG;
    for (const mesh of C.surf) {
      mesh.geometry.dispose();
      mesh.geometry = dphi > 0 && mesh.userData.region.loops.length ? revolveGeometry(mesh.userData.region.loops, state.phi0, dphi, opts.segments) : new THREE.BufferGeometry();
    }
    for (const mesh of C.curves3) {
      if (mesh.userData.kind !== 'curve') continue;
      mesh.geometry.dispose();
      mesh.geometry = dphi > 0 && mesh.userData.curve.S.length >= 4 ? revolveSegments(mesh.userData.curve.S, state.phi0, dphi, opts.segments) : new THREE.BufferGeometry();
    }
    const phis = [state.phi0, state.phi0 + dphi];
    C.planes.forEach((g, k) => { g.matrix.copy(planeBasis(phis[k])); g.matrixWorldNeedsUpdate = true; });
    layoutAxes3();
    updateClipping();
    dirty = true;
  }
  function updateClipping() {
    const m = model, L = m.L;
    const win3 = state.beyond ? m.view : { r_max: m.extent.r_max + 0.03 * L, z_min: m.extent.z_min - 0.03 * L, z_max: m.extent.z_max + 0.03 * L };
    const dphi = (360 - state.cutaway) * DEG;
    const phis = [state.phi0, state.phi0 + dphi];
    C.planes.forEach((g, k) => {
      const u = new THREE.Vector3(Math.cos(phis[k]), Math.sin(phis[k]), 0);
      const planes = [
        new THREE.Plane(u.clone().negate(), win3.r_max),
        new THREE.Plane(new THREE.Vector3(0, 0, 1), -win3.z_min),
        new THREE.Plane(new THREE.Vector3(0, 0, -1), win3.z_max),
      ];
      for (const ch of g.children) if (ch.material && (ch.userData.kind === 'field' || ch.isLineSegments)) ch.material.clippingPlanes = planes;
    });
    C.clip3 = win3;
  }
  let mapping = null;
  function currentStats() { return model.field && state.quantity ? model.field.stats[state.quantity] : null; }
  function cmapName() {
    if (state.colormap && state.colormap !== 'auto' && LUT_HEX[state.colormap]) return state.colormap;
    const st = currentStats(), q = state.quantity || '';
    if (!st) return 'viridis';
    if (state.scale !== 'log' && (st.signed || (qmeta(q).kind === 'component' && st.min < 0))) return 'coolwarm';
    if (qmeta(q).kind === 'flux') return 'cividis';
    return qmeta(q).family === 'pickup' ? 'plasma' : 'viridis';
  }
  let uvals = null;
  function updateField() {
    const f = model.field;
    if (!f || !state.quantity) { mapping = null; return; }
    const st = f.stats[state.quantity], vals = f.q[state.quantity];
    mapping = makeMapping(st, state.quantity, state.scale, state.decades, state.range);
    const n = vals.length;
    if (!uvals || uvals.length !== n) uvals = new Float32Array(n);
    for (let i = 0; i < n; i++) uvals[i] = mapping.map(vals[i]);
    if (C.fieldGeom) {
      const uv = C.fieldGeom.attributes.uv.array;
      for (let i = 0; i < n; i++) { uv[2 * i] = (0.5 + 255 * uvals[i]) / 256; uv[2 * i + 1] = 0.5; }
      C.fieldGeom.attributes.uv.needsUpdate = true;
    }
    lutData.set(lutTextureData(cmapName()));
    lutTex.needsUpdate = true;
    C.lineColors = { static: null, pickup: null };
    updateLineColors();
    drawColorbar();
    dirty = true;
  }
  function lineSamples(fam) {
    if (C.lineSamples[fam]) return C.lineSamples[fam];
    const grid = C.sampler.fieldGrid();
    const S = model.lines[fam], n = S.length >> 1, tri = new Int32Array(n), w = new Float32Array(3 * n), b = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      const t = grid ? grid.locate(S[2 * i], S[2 * i + 1], b) : -1;
      tri[i] = t; w[3 * i] = b[0]; w[3 * i + 1] = b[1]; w[3 * i + 2] = b[2];
    }
    return (C.lineSamples[fam] = { tri, w });
  }
  /** Per-vertex line colours: light over dark parts of the colormap, dark over bright parts. */
  function updateLineColors() {
    const f = model.field;
    for (const fam of ['static', 'pickup']) {
      const geom = C.lineGeom[fam];
      if (!geom || C.lineColors[fam] || !lineFamilyVisible(fam)) continue;
      const n = model.lines[fam].length >> 1, col = new Float32Array(3 * n);
      const [lt, dk] = LINE_TONES[fam].map(hexToRgb);
      if (f && mapping && uvals && layerVisible('field')) {
        const { tri, w } = lineSamples(fam), T = f.T, lb = lutBytes(cmapName());
        const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
        const lum = new Float32Array(256);
        for (let k = 0; k < 256; k++) lum[k] = 0.2126 * lin(lb[3 * k]) + 0.7152 * lin(lb[3 * k + 1]) + 0.0722 * lin(lb[3 * k + 2]);
        for (let i = 0; i < n; i++) {
          const t = tri[i];
          let u = 0;
          if (t >= 0) u = w[3 * i] * uvals[T[3 * t]] + w[3 * i + 1] * uvals[T[3 * t + 1]] + w[3 * i + 2] * uvals[T[3 * t + 2]];
          const y = lum[clamp(Math.round(255 * u), 0, 255)];
          const sm = clamp((y - 0.2) / 0.16, 0, 1), a = sm * sm * (3 - 2 * sm);
          for (let k = 0; k < 3; k++) col[3 * i + k] = lt[k] * (1 - a) + dk[k] * a;
        }
      } else {
        const bg = effectiveTheme() === 'light' ? dk : lt;
        for (let i = 0; i < n; i++) { col[3 * i] = bg[0]; col[3 * i + 1] = bg[1]; col[3 * i + 2] = bg[2]; }
      }
      geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
      C.lineColors[fam] = col;
    }
    dirty = true;
  }
  const lineFamilyVisible = (fam) => (state.linesAuto ? qmeta(state.quantity || '').family === fam && model.lines[fam].length > 0 : !state.hidden.has('lines:' + fam));
  function layerVisible(key) {
    if (key === 'lines:static' || key === 'lines:pickup') return lineFamilyVisible(key.slice(6));
    return !state.hidden.has(key);
  }
  function applyVisibility() {
    const c = C;
    const vis = (k) => layerVisible(k);
    const regVis = (R) => vis('region:' + R.name);
    const dphi = 360 - state.cutaway;
    c.group3.visible = state.mode === '3d';
    c.group2.visible = state.mode === 'section';
    for (const s of c.surf) s.visible = vis('surfaces') && regVis(s.userData.region) && dphi > 0;
    for (const s of c.curves3) {
      const cv = s.userData.curve;
      s.visible = vis('curve:' + cv.name) && (s.userData.kind === 'curve' ? vis('surfaces') : true);
    }
    for (const g of c.planes) g.visible = vis('cuts') && state.cutaway > 0;
    for (const f of c.field3) f.visible = vis('field') && !!mapping;
    for (const f of c.faces3) f.visible = regVis(f.userData.region);
    for (const o of c.outlines3) o.visible = regVis(o.userData.region);
    for (const fam of ['static', 'pickup']) { for (const l of c.lines3[fam]) l.visible = lineFamilyVisible(fam); for (const l of c.lines2[fam]) l.visible = lineFamilyVisible(fam); }
    c.axes3.visible = vis('axes');
    for (const f of c.field2) f.visible = vis('field') && !!mapping;
    for (const r of c.regions2) r.visible = regVis(r.userData.region);
    for (const o of c.outlines2) o.visible = regVis(o.userData.region);
    for (const s of c.curves2) s.visible = vis('curve:' + s.userData.curve.name);
    c.axis2.visible = vis('axes');
    C.lineColors = { static: null, pickup: null };
    updateLineColors();
    dirty = true;
  }
  function applyTransparency() {
    for (const s of C.surf) {
      const mat = s.material, base = mat.userData.baseOpacity ?? 1;
      const t = state.transparent || base < 1;
      mat.transparent = t;
      mat.opacity = state.transparent ? Math.min(base, 0.26) : base;
      mat.depthWrite = !t;
      mat.needsUpdate = true;
    }
    dirty = true;
  }

  /* -------------------------------------------------------------- theme */
  const mql = win.matchMedia ? win.matchMedia('(prefers-color-scheme: dark)') : null;
  let theme = THEMES.dark;
  function effectiveTheme() {
    if (state.theme === 'dark' || state.theme === 'light') return state.theme;
    return mql && !mql.matches ? 'light' : 'dark';
  }
  function applyTheme() {
    const t = effectiveTheme();
    root.dataset.theme = t;
    theme = THEMES[t];
    axisMat.color.set(theme.axis); axisMat2.color.set(theme.axis); coneMat.color.set(theme.axis);
    if (C) { C.lineColors = { static: null, pickup: null }; updateLineColors(); }
    drawColorbar();
    dirty = true;
  }
  if (mql) on(mql, 'change', () => { if (state.theme === 'auto') applyTheme(); });

  /* ----------------------------------------------------------------- UI */
  let ui = {};
  function buildUI() {
    ui = {};
    if (!opts.ui) {
      ui.cbar = h('div', { class: 'axf-cbar axf-glass' }, (ui.cbarCanvas = h('canvas')));
      root.append(ui.cbar, (ui.probe = h('div', { class: 'axf-probe', hidden: true })));
      return;
    }
    // header
    ui.name = h('div', { class: 'axf-name' });
    ui.sub = h('div', { class: 'axf-sub' });
    ui.badge = h('span', { class: 'axf-badge axf-glass', hidden: true });
    ui.mode3 = h('button', { type: 'button', 'aria-pressed': 'false', title: webgl ? '3D cutaway (revolved about the z axis)' : 'WebGL is unavailable in this browser', onclick: () => api.setMode('3d') }, '3D');
    if (!webgl) ui.mode3.disabled = true;
    ui.mode2 = h('button', { type: 'button', 'aria-pressed': 'false', title: 'Meridional (r, z) section, mirrored to r < 0', onclick: () => api.setMode('section') }, 'Section');
    ui.btnCtl = h('button', { type: 'button', class: 'axf-btn axf-glass', title: 'Display controls', 'aria-pressed': 'false', html: ICONS.sliders, onclick: () => togglePanel('controls') });
    ui.btnRes = h('button', { type: 'button', class: 'axf-btn axf-glass', title: 'Results', 'aria-pressed': 'false', html: ICONS.results, onclick: () => togglePanel('results') });
    ui.header = h('div', { class: 'axf-header' },
      h('div', { class: 'axf-title axf-glass' }, ui.name, ui.sub),
      h('div', { class: 'axf-tools' },
        ui.badge,
        h('div', { class: 'axf-seg axf-glass', role: 'group', 'aria-label': 'View mode' }, ui.mode3, ui.mode2),
        h('button', { type: 'button', class: 'axf-btn axf-glass', title: 'Reset view', html: ICONS.reset, onclick: () => api.resetView() }),
        h('button', { type: 'button', class: 'axf-btn axf-glass', title: 'Save PNG', html: ICONS.camera, onclick: savePNG }),
        ui.btnCtl, ui.btnRes));
    // controls panel
    ui.controls = h('div', { class: 'axf-panel axf-controls axf-glass', role: 'region', 'aria-label': 'Display controls' });
    ui.results = h('div', { class: 'axf-panel axf-results axf-res axf-glass', role: 'region', 'aria-label': 'Results' });
    ui.legend = h('div', { class: 'axf-legend axf-glass' });
    ui.cbar = h('div', { class: 'axf-cbar axf-glass', title: 'Colour scale' }, (ui.cbarCanvas = h('canvas')));
    ui.probe = h('div', { class: 'axf-probe', hidden: true });
    ui.msg = h('div', { class: 'axf-msg axf-glass', hidden: true, title: 'Click to dismiss', onclick: () => { ui.msg.hidden = true; } });
    root.append(ui.header, ui.controls, ui.results, ui.legend, ui.cbar, ui.msg, ui.probe);
    fillHeader(); fillControls(); fillResults(); fillLegend();
  }
  function fillHeader() {
    if (!ui.name) return;
    const title = opts.title || model.name;
    const sub = model.description ? model.description.split(/(?<=\.)\s/)[0] : '';
    ui.name.textContent = '';
    ui.name.append(title, h('small', null, model.field ? 'axisymmetric \u00b7 solved' : 'axisymmetric'));
    ui.name.title = model.description || title;
    ui.sub.textContent = sub;
    ui.sub.title = model.description || '';
    ui.sub.hidden = !sub;
  }
  function togglePanel(which, force) {
    const open = force ?? !state.panels[which];
    state.panels[which] = open;
    if (root.classList.contains('narrow') && open) state.panels[which === 'controls' ? 'results' : 'controls'] = false;
    syncPanels();
    if (!userMoved[state.mode]) fitView(); else updateViewOffset();
  }
  function syncPanels() {
    if (!ui.controls) return;
    ui.controls.hidden = !state.panels.controls;
    ui.results.hidden = !state.panels.results;
    ui.btnCtl.setAttribute('aria-pressed', String(state.panels.controls));
    ui.btnRes.setAttribute('aria-pressed', String(state.panels.results));
  }
  function seg(options, current, onPick) {
    const wrap = h('div', { class: 'axf-seg', role: 'group' });
    for (const [val, label] of options) {
      wrap.append(h('button', { type: 'button', 'data-v': val, 'aria-pressed': String(val === current), onclick: () => onPick(val) }, label));
    }
    return wrap;
  }
  function fillControls() {
    const el = ui.controls, f = model.field;
    el.textContent = '';
    el.append(h('div', { class: 'axf-ph' }, 'Display', h('button', { type: 'button', title: 'Close', onclick: () => togglePanel('controls', false) }, '×')));
    if (f && f.names.length) {
      ui.qsel = h('select', { class: 'axf-select', 'aria-label': 'Field quantity', onchange: (e) => api.setQuantity(e.target.value) });
      let fam = null, og = null;
      for (const q of f.names) {
        const meta = qmeta(q);
        if (meta.family !== fam) { fam = meta.family; og = h('optgroup', { label: fam === 'pickup' ? 'Pickup mode (per ampere)' : 'Static field B_lab' }); ui.qsel.append(og); }
        const st = f.stats[q];
        og.append(h('option', { value: q, disabled: st.zero ? true : null }, `${meta.text}  — ${meta.name}${st.zero ? ' (≡ 0)' : ''}`));
      }
      ui.qhint = h('div', { class: 'axf-hint' });
      el.append(h('div', { class: 'axf-field' }, h('div', { class: 'axf-lbl' }, 'Colour by'), ui.qsel, ui.qhint));
      ui.scaleSeg = seg([['linear', 'Linear'], ['log', 'Log']], state.scale, (v) => api.setScale(v));
      el.append(h('div', { class: 'axf-row' }, h('div', { class: 'axf-lbl' }, 'Scale'), ui.scaleSeg));
      ui.csel = h('select', { class: 'axf-select', style: { width: '128px' }, 'aria-label': 'Colormap', onchange: (e) => api.setColormap(e.target.value) },
        h('option', { value: 'auto' }, 'auto'), COLORMAPS.map((n) => h('option', { value: n }, n)));
      el.append(h('div', { class: 'axf-row' }, h('div', { class: 'axf-lbl' }, 'Colormap'), ui.csel));
    } else {
      el.append(h('div', { class: 'axf-note' }, 'This scene has no sampled field.'));
    }
    ui.only3 = h('div', null);
    ui.cutVal = h('span', null, `${Math.round(state.cutaway)}°`);
    ui.cut = h('input', { type: 'range', class: 'axf-range', min: '0', max: '300', step: '5', value: String(state.cutaway), 'aria-label': 'Cutaway angle', oninput: (e) => api.setCutaway(parseFloat(e.target.value)) });
    ui.only3.append(h('div', { class: 'axf-field' }, h('div', { class: 'axf-lbl' }, 'Cutaway wedge', ui.cutVal), ui.cut));
    ui.chkTransp = h('input', { type: 'checkbox', onchange: (e) => api.setTransparent(e.target.checked) });
    ui.chkBeyond = h('input', { type: 'checkbox', onchange: (e) => api.setFieldBeyondDevice(e.target.checked) });
    ui.only3.append(h('label', { class: 'axf-check' }, ui.chkTransp, 'Transparent surfaces'));
    if (f) ui.only3.append(h('label', { class: 'axf-check', title: 'Show the sampled field window beyond the device envelope on the cut planes' }, ui.chkBeyond, 'Field beyond the device'));
    el.append(ui.only3, h('div', { class: 'axf-sep' }), h('div', { class: 'axf-ph' }, 'Layers'));
    ui.layerChecks = new Map();
    const addLayer = (key, label, swatch, opts2 = {}) => {
      const input = h('input', { type: 'checkbox', disabled: opts2.disabled ? true : null, onchange: (e) => api.toggleLayer(key, e.target.checked) });
      ui.layerChecks.set(key, input);
      el.append(h('label', { class: 'axf-check' + (opts2.disabled ? ' dis' : ''), title: opts2.title || null }, input, swatch, h('span', null, label), opts2.note ? h('small', null, opts2.note) : null));
    };
    if (f) addLayer('field', 'Field map', h('span', { class: 'axf-sw', style: { background: 'linear-gradient(90deg,#440154,#21918c,#fde725)' } }));
    const torOnly = f && f.stats.Bphi && !f.stats.Bphi.zero && f.stats.psi && f.stats.psi.zero;
    addLayer('lines:static', 'Static field lines', lineSw(LINE_COLORS.static),
      model.lines.static.length ? { title: 'Contours of ψ = rAφ: exact poloidal field lines of B_lab' } : { disabled: true, note: torOnly ? 'B_lab is toroidal' : 'none', title: torOnly ? 'The static field is purely azimuthal (B_φ); it has no poloidal field lines.' : 'No static field lines in this scene.' });
    addLayer('lines:pickup', 'Pickup field lines', lineSw(LINE_COLORS.pickup),
      model.lines.pickup.length ? { title: 'Contours of the pickup flux function: poloidal field lines of the pickup mode' } : { disabled: true, note: 'none', title: 'No poloidal pickup field lines (the pickup field is azimuthal or absent).' });
    if (webgl) {
      addLayer('surfaces', 'Revolved surfaces', h('span', { class: 'axf-sw', style: { background: ROLE.superconductor.color } }), { note: '3D' });
      addLayer('cuts', 'Cut planes', h('span', { class: 'axf-sw', style: { background: 'linear-gradient(135deg,#440154 0 50%,#9aa4b2 50% 100%)' } }), { note: '3D' });
    }
    addLayer('axes', 'Axes and scale', lineSw('#c9d1d9'));
    el.append(h('div', { class: 'axf-sep' }), h('div', { class: 'axf-ph' }, 'Regions'));
    for (const R of model.regions) {
      const st = roleStyle(R.role);
      addLayer('region:' + R.name, R.name, h('span', { class: 'axf-sw', style: { background: R.role === 'superconductor' ? st.color : R.color } }),
        { note: R.role + (R.material && R.material !== R.role ? ` · ${R.material}` : '') });
    }
    for (const cv of model.curves) addLayer('curve:' + cv.name, cv.name, lineSw(ROLE.superconductor.color), { note: 'sheet' });
  }
  function syncControls() {
    if (!ui.controls) return;
    if (ui.mode3) { ui.mode3.setAttribute('aria-pressed', String(state.mode === '3d')); ui.mode2.setAttribute('aria-pressed', String(state.mode === 'section')); }
    if (ui.qsel && state.quantity) {
      ui.qsel.value = state.quantity;
      const st = currentStats();
      ui.qhint.textContent = qmeta(state.quantity).desc + (st && st.zero ? ' Identically zero in this scene.' : '');
    }
    if (ui.scaleSeg) for (const b of ui.scaleSeg.children) b.setAttribute('aria-pressed', String(b.dataset.v === state.scale));
    if (ui.csel) ui.csel.value = LUT_HEX[state.colormap] ? state.colormap : 'auto';
    if (ui.only3) ui.only3.style.display = state.mode === '3d' ? '' : 'none';
    if (ui.cut) { ui.cut.value = String(state.cutaway); ui.cutVal.textContent = `${Math.round(state.cutaway)}°`; }
    if (ui.chkTransp) ui.chkTransp.checked = state.transparent;
    if (ui.chkBeyond) ui.chkBeyond.checked = state.beyond;
    for (const [key, input] of ui.layerChecks || []) input.checked = !input.disabled && layerVisible(key);
    fillLegend();
  }
  function fillLegend() {
    if (!ui.legend) return;
    const el = ui.legend;
    el.textContent = '';
    const roles = new Map();
    for (const R of model.regions) if (!roles.has(R.role)) roles.set(R.role, R);
    const row = h('div');
    for (const [role, R] of roles) {
      const st = roleStyle(role);
      const mats = [...new Set(model.regions.filter((x) => x.role === role).map((x) => x.material).filter((m) => m && m !== role))];
      row.append(h('span', { class: 'it' }, h('span', { class: 'axf-sw', style: { background: role === 'superconductor' ? st.color : R.color } }), st.label + (mats.length ? ` (${mats.join(', ')})` : '')));
    }
    if (model.curves.length) row.append(h('span', { class: 'it' }, lineSw(ROLE.superconductor.color), 'superconducting sheet'));
    for (const fam of ['static', 'pickup']) {
      if (lineFamilyVisible(fam)) row.append(h('span', { class: 'it' }, lineSw(LINE_COLORS[fam]), fam === 'static' ? 'field lines of B_lab' : 'pickup field lines'));
    }
    el.append(row);
    const notes = [];
    if (model.field && state.quantity) {
      const meta = qmeta(state.quantity);
      notes.push(meta.family === 'pickup' ? 'Colour: field of a unit readout current (per ampere).' : 'Colour: static laboratory field B_lab.');
    }
    if (lineFamilyVisible('static') || lineFamilyVisible('pickup')) notes.push('Lines: contours of ψ = rA_φ (exact poloidal field lines).');
    if (model.regions.some((R) => R.role === 'superconductor')) notes.push('B = 0 inside superconductors.');
    if (notes.length) el.append(h('div', { class: 'ln' }, notes.join(' ')));
  }

  /* ------------------------------------------------------------ results */
  function fillResults() {
    if (!ui.results) return;
    const el = ui.results, S = model.summary || {};
    el.textContent = '';
    el.append(h('div', { class: 'axf-ph' }, 'Results', h('button', { type: 'button', title: 'Close', onclick: () => togglePanel('results', false) }, '×')));
    const kv = () => h('div', { class: 'axf-kv' });
    const row = (g, k, v, sub, title) => {
      g.append(h('div', { class: 'k', title: title || null, html: k }), h('div', { class: 'v', html: v }));
      if (sub) g.append(h('div', { class: 's', html: sub }));
    };
    const sec = (title, open, body) => h('details', { open: open ? true : null }, h('summary', null, title), body);
    const regionRole = (name) => model.regions.find((R) => R.name === name)?.role;
    const B = (x) => esc(fmtVal(x, 'T'));
    let any = false;
    const cpl = S.coupling;
    if (cpl) {
      any = true;
      const g = kv();
      row(g, 'Ψ <span class="s">signal coupling</span>', `${esc(fmtNum(cpl.Psi_SI_H_T_m2))} H·T·m²`,
        Number.isFinite(cpl.Psi_natural_T_m3) ? `${esc(fmtNum(cpl.Psi_natural_T_m3))} T·m³ (natural)` : '',
        'Ψ = (1/I)∫A_I·B_lab d³x; the axion flux is Φ_a = κ_a Ψ');
      row(g, 'L<sub>p</sub> <span class="s">pickup inductance</span>', esc(fmtSI(cpl.L_p_H, 'H')),
        Number.isFinite(cpl.L_p_natural_m) ? `${esc(fmtNum(cpl.L_p_natural_m))} m (natural)` : '');
      row(g, '<span class="axf-cal">F</span> = Ψ⁴/L<sub>p</sub>²', `${esc(fmtNum(cpl.F_natural_T4_m10))} T⁴·m¹⁰`,
        Number.isFinite(cpl.F_SI_T4_H2_m8) ? `${esc(fmtNum(cpl.F_SI_T4_H2_m8))} H²·T⁴·m⁸ (SI)` : '', 'Scan figure of merit; the requirement is F ≥ F_req');
      row(g, 'W<sub>mag</sub> <span class="s">stored energy</span>', esc(fmtSI(cpl.W_mag_J, 'J')));
      row(g, 'g <span class="s">geometry factor</span>', esc(fmtNum(cpl.g_factor)), '', cpl.note || 'g = Ψ²/(L_p W R²), natural units');
      if (Number.isFinite(cpl.device_radius_m)) row(g, 'Device R × height', `${esc(fmtNum(cpl.device_radius_m))} m × ${esc(fmtNum(cpl.device_height_m))} m`);
      el.append(sec('Signal coupling', true, g));
    }
    const fl = S.fields;
    if (fl) {
      any = true;
      const g = kv();
      const loc = fl.B_max_location;
      row(g, 'Peak |B|', B(fl.B_max_T), loc ? `at r = ${esc(fmtNum(loc.r_m))} m, z = ${esc(fmtNum(loc.z_m))} m${loc.region ? ` (${esc(loc.region)})` : ''}` : '');
      const by = fl.by_region_T || {};
      const wind = Object.keys(by).filter((k) => regionRole(k) === 'winding');
      if (wind.length) { g.append(h('div', { class: 'hd' }, 'Peak |B| in windings')); for (const k of wind) row(g, esc(k), B(by[k])); }
      const sc = fl.superconductor_surface_T || {};
      if (Object.keys(sc).length) { g.append(h('div', { class: 'hd' }, 'Peak |B| on superconductor surfaces')); for (const [k, v] of Object.entries(sc)) row(g, esc(k), B(v)); }
      const other = Object.keys(by).filter((k) => !wind.includes(k));
      if (other.length) { g.append(h('div', { class: 'hd' }, 'Peak |B| in other regions')); for (const k of other) row(g, esc(k === 'air' ? 'air (vacuum)' : k), B(by[k])); }
      el.append(sec('Peak fields', true, g));
    }
    if (Array.isArray(S.benchmarks) && S.benchmarks.length) {
      any = true;
      const wrap = h('div');
      wrap.append(h('div', { class: 'axf-note' }, 'Amplitude scale = (F_req/F)^1/4: the factor on all source currents needed to meet the requirement at the present size. Cost-optimal size: the design scaled uniformly to minimum cost at the requirement.'));
      for (const b of S.benchmarks) {
        const ok = b.F_over_F_req >= 1;
        const g = kv();
        row(g, '<span class="axf-cal">F</span><sub>req</sub>', `${esc(fmtNum(b.F_req_natural_T4_m10))} T⁴·m¹⁰`);
        row(g, '<span class="axf-cal">F</span>/<span class="axf-cal">F</span><sub>req</sub>', `<span class="axf-pill ${ok ? 'ok' : 'short'}">${esc(fmtNum(b.F_over_F_req))}</span>`);
        row(g, 'Amplitude scale', `×${esc(fmtNum(b.amplitude_scale))}`);
        g.append(h('div', { class: 'hd' }, 'At requirement, present size'));
        row(g, 'Peak |B|', B(b.B_max_at_req_T));
        row(g, 'W<sub>mag</sub>', esc(fmtSI(b.W_mag_at_req_J, 'J')));
        const scr = b.superconductor_surface_at_req_T || {};
        for (const [k, v] of Object.entries(scr)) row(g, `|B| on ${esc(k)} surface`, B(v));
        if (b.cost) row(g, 'Cost', `${esc(fmtNum(b.cost.total_MCHF))} MCHF`, costSplit(b.cost));
        const co = b.cost_optimal_size;
        if (co) {
          g.append(h('div', { class: 'hd' }, 'Cost-optimal size'));
          row(g, 'Device radius', `${esc(fmtNum(co.device_radius_m))} m`, Number.isFinite(co.length_scale) ? `length scale ×${esc(fmtNum(co.length_scale))}` : '');
          row(g, 'Peak |B|', B(co.B_max_T));
          row(g, 'W<sub>mag</sub>', esc(fmtSI(co.W_mag_J, 'J')));
          if (co.cost) row(g, 'Cost', `${esc(fmtNum(co.cost.total_MCHF))} MCHF`, costSplit(co.cost));
        }
        const meta = [];
        if (Number.isFinite(b.Q)) meta.push(`Q = ${fmtNum(b.Q, 3)}`);
        if (Number.isFinite(b.T_K)) meta.push(`T = ${fmtSI(b.T_K, 'K', 3)}`);
        if (Number.isFinite(b.eta_A)) meta.push(`η_A = ${fmtNum(b.eta_A, 3)}`);
        if (Number.isFinite(b.g_min_GeV)) meta.push(`g_min = ${fmtNum(b.g_min_GeV, 4)} GeV⁻¹`);
        wrap.append(h('div', { class: 'axf-bench' }, h('h4', null, esc(b.name || 'benchmark'), h('small', null, meta.join(' · '))), g));
      }
      el.append(sec('Benchmarks and cost', true, wrap));
    }
    if (S.forces && Object.keys(S.forces).length) {
      any = true;
      const g = kv();
      for (const [k, v] of Object.entries(S.forces)) {
        if (k === 'closure' || k === 'distributions') continue;  // rendered separately / not in the scene
        const name = k.replace(/^(superconductor|iron):/, '');
        const kind = k.startsWith('superconductor:') ? ' (superconductor)' : k.startsWith('iron:') ? ' (iron)' : '';
        g.append(h('div', { class: 'hd' }, name + kind));
        if (Number.isFinite(v.Fz_excluding_self_N)) {
          row(g, 'F<sub>z</sub> (axial)', esc(fmtSI(v.Fz_excluding_self_N, 'N')), '',
              'Axial force from all other sources; the winding’s own (spurious) self-force has been removed');
          row(g, 'F<sub>z</sub> self-force', esc(fmtSI(v.Fz_self_N, 'N')), esc(fmtSI(v.Fz_N, 'N')) + ' raw',
              'A loop exerts no net axial force on itself, so this is the discretization error: the error bar on F_z');
        } else if (Number.isFinite(v.Fz_N)) {
          row(g, 'F<sub>z</sub> (axial)', esc(fmtSI(v.Fz_N, 'N')));
          if (Number.isFinite(v.Fz_self_N)) {
            row(g, 'F<sub>z</sub> self-force', esc(fmtSI(v.Fz_self_N, 'N')), '',
                'Self-force residual: the error bar on F_z (not removed, because the design is nonlinear)');
          }
        }
        if (Number.isFinite(v.radial_force_per_radian_N)) row(g, 'F<sub>r</sub> per radian', esc(fmtSI(v.radial_force_per_radian_N, 'N/rad')), '', 'Net radial Lorentz force per radian of azimuth (positive outward)');
      }
      const cl = S.forces.closure;
      if (cl && Number.isFinite(cl.sum_Fz_N)) {
        g.append(h('div', { class: 'hd' }, 'Newton’s third law'));
        row(g, 'Σ F<sub>z</sub> (must vanish)', esc(fmtSI(cl.sum_Fz_N, 'N')),
            Number.isFinite(cl.relative) ? esc(fmtNum(cl.relative, 2)) + ' of the largest force' : '',
            'The device is isolated, so the axial forces on all its bodies must cancel; what is left measures the accuracy of the force post-processing');
        for (const c of cl.caveats || []) row(g, 'caveat', esc(c));
      }
      el.append(sec('Forces', false, g));
    }
    if (S.geometry) {
      any = true;
      const g = kv(), G = S.geometry;
      if (Number.isFinite(G.A_cold_m2)) row(g, 'Cold surface area', `${esc(fmtNum(G.A_cold_m2))} m²`);
      const ce = G.cold_envelope;
      if (ce) row(g, 'Cold envelope', `r ≤ ${esc(fmtNum(ce.r_max_m))} m`, `${esc(fmtNum(ce.z_min_m))} ≤ z ≤ ${esc(fmtNum(ce.z_max_m))} m`);
      if (Number.isFinite(G.iron_mass_kg)) row(g, 'Iron mass', `${esc(fmtNum(G.iron_mass_kg))} kg`);
      el.append(sec('Geometry', false, g));
    }
    if (S.pickup) {
      any = true;
      const g = kv(), P = S.pickup;
      row(g, 'Pickup', esc(`${P.kind || ''}${P.conductor ? ` · conductor ${P.conductor}` : ''}${P.port ? ` · port ${P.port}` : ''}`));
      if (Number.isFinite(P.L_p_H)) row(g, 'L<sub>p</sub> (energy)', esc(fmtSI(P.L_p_H, 'H', 6)));
      if (Number.isFinite(P.L_p_from_flux_H)) {
        const rel = Math.abs(P.L_p_from_flux_H - P.L_p_H) / Math.abs(P.L_p_H || 1);
        row(g, 'L<sub>p</sub> (flux)', esc(fmtSI(P.L_p_from_flux_H, 'H', 6)), `relative difference ${esc(fmtNum(rel, 2))}`, 'Independent consistency check: inductance from linked flux versus from field energy');
      }
      if (Number.isFinite(P.energy_J)) row(g, 'Energy at ' + esc(fmtNum(P.current_A ?? 1)) + ' A', esc(fmtSI(P.energy_J, 'J')));
      el.append(sec('Pickup mode', false, g));
    }
    const params = model.design?.parameters;
    if (params && typeof params === 'object' && Object.keys(params).length) {
      const g = kv();
      for (const [k, v] of Object.entries(params)) row(g, esc(k), esc(typeof v === 'number' ? fmtNum(v, 6) : String(v)));
      el.append(sec('Design parameters', false, h('div', null, model.description ? h('div', { class: 'axf-note' }, model.description) : null, g)));
    }
    if (!any) el.append(h('div', { class: 'axf-note' }, 'No results summary in this scene.'));
    if (model.warnings.length) el.append(h('div', { class: 'axf-note', style: { color: 'var(--warn)' } }, 'Scene warnings: ' + model.warnings.join('; ')));
    el.append(sec('Summary JSON', false, h('pre', null, JSON.stringify(S, null, 1))));
  }
  function costSplit(c) {
    const parts = [];
    if (Number.isFinite(c.magnet_MCHF)) parts.push(`magnet ${esc(fmtNum(c.magnet_MCHF))}`);
    if (Number.isFinite(c.cryostat_MCHF)) parts.push(`cryostat ${esc(fmtNum(c.cryostat_MCHF))}`);
    if (Number.isFinite(c.iron_MCHF) && c.iron_MCHF) parts.push(`iron ${esc(fmtNum(c.iron_MCHF))}`);
    return parts.join(' + ');
  }

  /* ----------------------------------------------------------- colorbar */
  const lutCanvas = h('canvas', { width: '256', height: '1' });
  function drawColorbar() {
    const cv = ui.cbarCanvas;
    if (!cv) return;
    const show = !!(mapping && model.field && layerVisible('field'));
    ui.cbar.style.display = show ? '' : 'none';
    if (!show) return;
    const cw = Math.max(120, ui.cbar.clientWidth || 320), ch = Math.max(40, ui.cbar.clientHeight || 74);
    cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr);
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    // LUT strip
    const lctx = lutCanvas.getContext('2d'), img = lctx.createImageData(256, 1), lb = lutBytes(cmapName());
    for (let i = 0; i < 256; i++) { img.data[4 * i] = lb[3 * i]; img.data[4 * i + 1] = lb[3 * i + 1]; img.data[4 * i + 2] = lb[3 * i + 2]; img.data[4 * i + 3] = 255; }
    lctx.putImageData(img, 0, 0);
    const pad = 12, bx = pad, bw = cw - 2 * pad, by = 27, bh = 11;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(lutCanvas, 0, 0, 256, 1, bx, by, bw, bh);
    ctx.strokeStyle = 'rgba(127,127,127,0.45)'; ctx.lineWidth = 1;
    ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
    // title
    const q = state.quantity, meta = qmeta(q), unit = model.field.units[q] || '';
    const mp = mapping;
    ctx.fillStyle = theme.fg; ctx.textBaseline = 'alphabetic';
    let parts = meta.sym;
    if (mp.log && mp.abs) parts = [['|'], ...parts, ['|']];
    let pre = { e3: 0, p: '' };
    if (!mp.log) pre = prefixFor(Math.max(Math.abs(mp.lo), Math.abs(mp.hi)));
    const unitTxt = unit ? ` [${pre.p}${unit}]` : '';
    let x = drawRich(ctx, parts, bx, 18, 12.5, 650);
    ctx.font = `500 11.5px ${FONT}`; ctx.fillStyle = theme.fg2;
    const fam = meta.family === 'pickup' ? 'pickup, per A' : 'static B_lab';
    const tail = `${unitTxt}  · ${fam}${mp.log ? ' · log' : ''}${mp.uniform ? ' · uniform' : ''}`;
    ctx.fillText(tail, bx + x, 18);
    // ticks
    ctx.font = `500 10.5px ${FONT}`; ctx.fillStyle = theme.fg2; ctx.strokeStyle = theme.fg2; ctx.textAlign = 'center';
    const ty = by + bh, ticks = [];
    if (mp.log) {
      const d0 = Math.round(Math.log10(mp.lo)), d1 = Math.floor(Math.log10(mp.hi) + 1e-9);
      const nd = d1 - d0, every = nd > 8 ? 3 : nd > 5 ? 2 : 1;
      for (let d = d0; d <= d1; d++) ticks.push({ u: mp.map(10 ** d), t: (d - d0) % every === 0 ? [['10'], [minus(String(d)), 'sup']] : '', minor: false });
      if (nd <= 4) for (let d = d0; d < d1 + 1; d++) for (let k = 2; k < 10; k++) { const v = k * 10 ** d; if (v > mp.lo && v < mp.hi) ticks.push({ u: mp.map(v), t: '', minor: true }); }
    } else if (!mp.uniform) {
      const sc = 10 ** -pre.e3;
      const nt = niceTicks(mp.lo * sc, mp.hi * sc, Math.max(3, Math.floor(bw / 62)));
      for (const t of nt.ticks) ticks.push({ u: mp.map(t / sc), t: tickText(t, nt.step), minor: false });
    } else {
      ticks.push({ u: 0.5, t: fmtVal(model.field.stats[q].max, unit) });
    }
    for (const tk of ticks) {
      const tx = bx + tk.u * bw;
      ctx.beginPath(); ctx.moveTo(Math.round(tx) + 0.5, ty); ctx.lineTo(Math.round(tx) + 0.5, ty + (tk.minor ? 3 : 5)); ctx.stroke();
      if (Array.isArray(tk.t)) {
        const w = measureRich(ctx, tk.t, 11, 500);
        ctx.textAlign = 'left';
        drawRich(ctx, tk.t, clamp(tx, bx + 12, bx + bw - 12) - w / 2, ty + 18, 11, 500);
        ctx.font = `500 10.5px ${FONT}`; ctx.textAlign = 'center';
      } else if (tk.t) ctx.fillText(tk.t, clamp(tx, bx + 12, bx + bw - 12), ty + 18);
    }
    ctx.textAlign = 'left';
  }

  /* ------------------------------------------------------------ layout */
  function freeRect() {
    const rr = root.getBoundingClientRect();
    let l = 0, r = W, t = 0, b = H;
    if (opts.ui && ui.header) {
      const hb = ui.header.getBoundingClientRect();
      t = Math.max(0, hb.bottom - rr.top + 4);
      const narrow = root.classList.contains('narrow');
      if (!narrow) {
        if (state.panels.controls && !ui.controls.hidden) l = ui.controls.getBoundingClientRect().right - rr.left + 6;
        if (state.panels.results && !ui.results.hidden) r = ui.results.getBoundingClientRect().left - rr.left - 6;
        const lg = ui.legend.getBoundingClientRect(), cb = ui.cbar.getBoundingClientRect();
        b = H - Math.max(lg.height, cb.height) - 18;
      } else if (ui.cbar.style.display !== 'none') {
        b = ui.cbar.getBoundingClientRect().top - rr.top - 6;
      }
    }
    if (r - l < 0.4 * W) { l = 0; r = W; }
    if (b - t < 0.4 * H) { t = 0; b = H; }
    return { l, r, t, b };
  }
  function updateViewOffset() {
    if (!webgl) { dirty = true; return; }
    const fr = freeRect();
    const sx = (fr.l + fr.r) / 2 - W / 2, sy = (fr.t + fr.b) / 2 - H / 2;
    const fw = W + 2 * Math.abs(sx), fh = H + 2 * Math.abs(sy);
    cam3.aspect = fw / fh;
    if (Math.abs(sx) < 0.5 && Math.abs(sy) < 0.5) cam3.clearViewOffset();
    else cam3.setViewOffset(fw, fh, Math.abs(sx) - sx, Math.abs(sy) - sy, W, H);
    cam3.updateProjectionMatrix();
    dirty = true;
    return fr;
  }
  let ppu2 = 100; // section view: CSS pixels per metre at zoom 1
  const view2 = { cx: 0, cy: 0, zoom: 1 }; // canvas fallback camera
  function fit3D() {
    const fr = updateViewOffset();
    const e = model.extent;
    const zc = (e.z_min + e.z_max) / 2, hz = (e.z_max - e.z_min) / 2;
    const rad = Math.hypot(e.r_max, hz) * 1.04;
    const vf = cam3.fov * DEG;
    const fw = fr.r - fr.l, fh = fr.b - fr.t;
    const fullH = H + 2 * Math.abs((fr.t + fr.b) / 2 - H / 2);
    const tv = Math.tan(vf / 2) * (fh / fullH), th = Math.tan(vf / 2) * (fw / fullH);
    const d = rad / Math.sin(Math.atan(Math.min(tv, th)));
    const az = state.phi0 - (state.cutaway * DEG) / 2, el = clamp(opts.elevation, -89, 89) * DEG;
    cam3.position.set(d * Math.cos(el) * Math.cos(az), d * Math.cos(el) * Math.sin(az), zc + d * Math.sin(el));
    cam3.near = d / 200; cam3.far = d * 20;
    cam3.updateProjectionMatrix();
    controls3.target.set(0, 0, zc);
    controls3.minDistance = d * 0.05; controls3.maxDistance = d * 8;
    controls3.update();
    userMoved['3d'] = false;
    dirty = true;
  }
  function fit2D() {
    const fr = freeRect();
    const v = model.view, mL = 50, mB = 36, mT = 12, mR = 16;
    const aw = Math.max(40, fr.r - fr.l - mL - mR), ah = Math.max(40, fr.b - fr.t - mT - mB);
    const wx = 2 * v.r_max, wy = v.z_max - v.z_min;
    ppu2 = Math.min(aw / wx, ah / wy);
    const cxp = fr.l + mL + aw / 2, cyp = fr.t + mT + ah / 2, zc = (v.z_min + v.z_max) / 2;
    const cx = -(cxp - W / 2) / ppu2, cy = zc + (cyp - H / 2) / ppu2;
    if (webgl) {
      cam2.left = -W / 2 / ppu2; cam2.right = W / 2 / ppu2; cam2.top = H / 2 / ppu2; cam2.bottom = -H / 2 / ppu2;
      cam2.zoom = 1;
      cam2.position.set(cx, cy, 10 * model.L);
      cam2.near = 0.01 * model.L; cam2.far = 100 * model.L;
      cam2.updateProjectionMatrix();
      controls2.target.set(cx, cy, 0);
      controls2.minZoom = 0.2; controls2.maxZoom = 400;
      controls2.update();
    } else {
      view2.cx = cx; view2.cy = cy; view2.zoom = 1;
    }
    userMoved.section = false;
    dirty = true;
  }
  function fitView() { if (state.mode === '3d' && webgl) fit3D(); else fit2D(); }
  function map2() {
    const ppu = webgl ? (W * cam2.zoom) / (cam2.right - cam2.left) : ppu2 * view2.zoom;
    const cx = webgl ? cam2.position.x : view2.cx, cy = webgl ? cam2.position.y : view2.cy;
    return { ppu, cx, cy, toPx: (x, y) => [W / 2 + (x - cx) * ppu, H / 2 - (y - cy) * ppu], toWorld: (px, py) => [cx + (px - W / 2) / ppu, cy - (py - H / 2) / ppu] };
  }
  function resize() {
    const w = Math.max(1, root.clientWidth), hh = Math.max(1, root.clientHeight);
    const changed = w !== W || hh !== H;
    W = w; H = hh;
    dpr = Math.min(win.devicePixelRatio || 1, opts.maxPixelRatio);
    root.classList.toggle('narrow', W < 760);
    if (root.classList.contains('narrow') && state.panels.controls && state.panels.results) { state.panels.results = false; syncPanels(); }
    if (webgl) { renderer.setPixelRatio(dpr); renderer.setSize(W, H, false); }
    else { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
    overlay.width = Math.round(W * dpr); overlay.height = Math.round(H * dpr);
    drawColorbar();
    if (webgl) {
      updateViewOffset();
      const p = ppu2;
      cam2.left = -W / 2 / p; cam2.right = W / 2 / p; cam2.top = H / 2 / p; cam2.bottom = -H / 2 / p;
      cam2.updateProjectionMatrix();
    }
    if (changed) { if (!userMoved['3d'] && webgl) fit3D(); if (!userMoved.section) fit2D(); }
    dirty = true;
  }

  /* ---------------------------------------------------------- rendering */
  let dirty = true, raf = 0, disposed = false;
  function renderNow() {
    if (disposed) return;
    if (webgl) {
      if (state.mode === '3d') renderer.render(scene3, cam3);
      else renderer.render(scene2, cam2);
    } else {
      renderCanvas2D();
    }
    drawOverlay();
    dirty = false;
  }
  function loop() {
    raf = win.requestAnimationFrame(loop);
    if (webgl) {
      const c = state.mode === '3d' ? controls3 : controls2;
      if (c.update()) dirty = true;
    }
    if (dirty) renderNow();
    if (probeReq) doProbe();
  }
  function haloText(ctx, text, x, y, align = 'center', bold = false) {
    ctx.font = `${bold ? 700 : 500} ${bold ? 13 : 11}px ${FONT}`;
    ctx.textAlign = align;
    ctx.lineWidth = 3; ctx.strokeStyle = theme.halo; ctx.lineJoin = 'round';
    ctx.strokeText(text, x, y);
    ctx.fillStyle = theme.fg; ctx.fillText(text, x, y);
  }
  function drawOverlay() {
    const ctx = octx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (pinned && ui.probe && !ui.probe.hidden) {
      ctx.beginPath(); ctx.arc(pinned.x, pinned.y, 5.5, 0, 2 * Math.PI);
      ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(0,0,0,0.55)'; ctx.stroke();
      ctx.lineWidth = 1.5; ctx.strokeStyle = '#ffffff'; ctx.stroke();
    }
    if (!layerVisible('axes')) return;
    if (state.mode === '3d' && webgl) {
      cam3.updateMatrixWorld();
      const v = new THREE.Vector3();
      ctx.textBaseline = 'middle';
      for (const lb of C.labels3 || []) {
        v.copy(lb.p).project(cam3);
        if (v.z > 1 || v.z < -1) continue;
        haloText(ctx, lb.text, ((v.x + 1) / 2) * W, ((1 - v.y) / 2) * H, lb.align || 'center', lb.bold);
      }
    } else {
      drawSectionAxes(ctx);
    }
  }
  function drawSectionAxes(ctx) {
    const mp = map2(), fr = freeRect();
    const x0 = fr.l + 50, y1 = fr.b - 34, x1 = fr.r - 12, y0 = fr.t + 8;
    if (x1 - x0 < 60 || y1 - y0 < 60) return;
    const [wl, wt] = mp.toWorld(x0, y0), [wr, wb] = mp.toWorld(x1, y1);
    ctx.strokeStyle = theme.fg2; ctx.lineWidth = 1; ctx.fillStyle = theme.fg2;
    ctx.textBaseline = 'top';
    // r ticks (bottom), labelled |x| since the left half is the mirror image
    const xt = niceTicks(wl, wr, Math.max(3, Math.floor((x1 - x0) / 80)));
    ctx.beginPath(); ctx.moveTo(x0, y1 + 0.5); ctx.lineTo(x1, y1 + 0.5); ctx.stroke();
    for (const t of xt.ticks) {
      const [px] = mp.toPx(t, 0);
      ctx.beginPath(); ctx.moveTo(Math.round(px) + 0.5, y1); ctx.lineTo(Math.round(px) + 0.5, y1 + 5); ctx.stroke();
      haloText(ctx, tickText(Math.abs(t), xt.step), px, y1 + 13);
    }
    haloText(ctx, 'r (m)', x1 - 16, y1 + 27, 'right', true);
    // z ticks (left)
    const zt = niceTicks(wb, wt, Math.max(3, Math.floor((y1 - y0) / 60)));
    ctx.beginPath(); ctx.moveTo(x0 - 0.5, y0); ctx.lineTo(x0 - 0.5, y1); ctx.stroke();
    for (const t of zt.ticks) {
      const py = mp.toPx(0, t)[1];
      ctx.beginPath(); ctx.moveTo(x0 - 5, Math.round(py) + 0.5); ctx.lineTo(x0, Math.round(py) + 0.5); ctx.stroke();
      ctx.textBaseline = 'middle';
      haloText(ctx, tickText(t, zt.step), x0 - 8, py, 'right');
    }
    ctx.textBaseline = 'middle';
    haloText(ctx, 'z (m)', x0 + 6, y0 + 8, 'left', true);
  }

  /* --------------------------------------------- Canvas2D section fallback */
  function renderCanvas2D() {
    const ctx = c2d, cw = canvas.width, ch = canvas.height, mp = map2();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    const m = model, f = m.field;
    const k = dpr;
    const X = (x) => (W / 2 + (x - mp.cx) * mp.ppu) * k, Y = (y) => (H / 2 - (y - mp.cy) * mp.ppu) * k;
    const rx0 = Math.max(0, Math.floor(X(-m.view.r_max))), rx1 = Math.min(cw - 1, Math.ceil(X(m.view.r_max)) - 1);
    const ry0 = Math.max(0, Math.floor(Y(m.view.z_max))), ry1 = Math.min(ch - 1, Math.ceil(Y(m.view.z_min)) - 1);
    if (f && mapping && layerVisible('field')) {
      const img = ctx.createImageData(cw, ch), D = img.data, lb = lutBytes(cmapName());
      const V = f.V, T = f.T;
      for (const side of [1, -1]) {
        for (let t = 0; t < T.length; t += 3) {
          const a = T[t], b = T[t + 1], c = T[t + 2];
          const xa = X(side * V[2 * a]), ya = Y(V[2 * a + 1]), xb = X(side * V[2 * b]), yb = Y(V[2 * b + 1]), xc = X(side * V[2 * c]), yc = Y(V[2 * c + 1]);
          const ix0 = Math.max(rx0, Math.floor(Math.min(xa, xb, xc))), ix1 = Math.min(rx1, Math.ceil(Math.max(xa, xb, xc)));
          const iy0 = Math.max(ry0, Math.floor(Math.min(ya, yb, yc))), iy1 = Math.min(ry1, Math.ceil(Math.max(ya, yb, yc)));
          if (ix0 > ix1 || iy0 > iy1) continue;
          const d = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc);
          if (d === 0) continue;
          const ua = uvals[a], ub = uvals[b], uc = uvals[c];
          for (let py = iy0; py <= iy1; py++) {
            const y = py + 0.5;
            for (let px = ix0; px <= ix1; px++) {
              const x = px + 0.5;
              const l0 = ((yb - yc) * (x - xc) + (xc - xb) * (y - yc)) / d;
              if (l0 < -1e-6) continue;
              const l1 = ((yc - ya) * (x - xc) + (xa - xc) * (y - yc)) / d;
              const l2 = 1 - l0 - l1;
              if (l1 < -1e-6 || l2 < -1e-6) continue;
              const li = Math.round(255 * (l0 * ua + l1 * ub + l2 * uc)) * 3, o = 4 * (py * cw + px);
              D[o] = lb[li]; D[o + 1] = lb[li + 1]; D[o + 2] = lb[li + 2]; D[o + 3] = 255;
            }
          }
        }
      }
      ctx.putImageData(img, 0, 0);
    }
    ctx.setTransform(k, 0, 0, k, 0, 0);
    const px = (x) => W / 2 + (x - mp.cx) * mp.ppu, py = (y) => H / 2 - (y - mp.cy) * mp.ppu;
    for (const R of m.regions) {
      if (!layerVisible('region:' + R.name)) continue;
      const st = roleStyle(R.role);
      for (const side of [1, -1]) {
        const p = new Path2D();
        for (let t = 0; t < R.T.length; t += 3) {
          const a = R.T[t], b = R.T[t + 1], c = R.T[t + 2];
          p.moveTo(px(side * R.V[2 * a]), py(R.V[2 * a + 1])); p.lineTo(px(side * R.V[2 * b]), py(R.V[2 * b + 1])); p.lineTo(px(side * R.V[2 * c]), py(R.V[2 * c + 1])); p.closePath();
        }
        ctx.globalAlpha = st.alpha2d; ctx.fillStyle = R.role === 'superconductor' ? st.flat : R.color; ctx.fill(p);
        ctx.globalAlpha = 1;
        const o = new Path2D();
        for (const lp of R.loops) {
          const P = lp.pts, n = P.length >> 1;
          o.moveTo(px(side * P[0]), py(P[1]));
          for (let i = 1; i < n; i++) o.lineTo(px(side * P[2 * i]), py(P[2 * i + 1]));
          o.closePath();
        }
        ctx.strokeStyle = R.role === 'superconductor' ? '#dfe5ec' : mixHex(R.color, '#ffffff', 0.25); ctx.lineWidth = 1; ctx.stroke(o);
      }
    }
    const segs = (S, color, alpha, keep) => {
      ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = 1;
      const p = new Path2D();
      for (const side of [1, -1]) for (let i = 0; i + 3 < S.length; i += 4) { if (keep && !keep(i >> 1)) continue; p.moveTo(px(side * S[i]), py(S[i + 1])); p.lineTo(px(side * S[i + 2]), py(S[i + 3])); }
      ctx.stroke(p); ctx.globalAlpha = 1;
    };
    ctx.save();
    ctx.beginPath(); ctx.rect(px(-m.view.r_max), py(m.view.z_max), 2 * m.view.r_max * mp.ppu, (m.view.z_max - m.view.z_min) * mp.ppu); ctx.clip();
    for (const fam of ['static', 'pickup']) {
      if (!lineFamilyVisible(fam)) continue;
      const col = C.lineColors[fam], tones = LINE_TONES[fam];
      if (!col) { segs(m.lines[fam], tones[0], 0.75); continue; }
      const dark = (v) => col[3 * v] + col[3 * v + 1] + col[3 * v + 2] < 1.5;
      segs(m.lines[fam], tones[0], 0.75, (v) => !dark(v));
      segs(m.lines[fam], tones[1], 0.75, (v) => dark(v));
    }
    ctx.restore();
    for (const cv of m.curves) if (layerVisible('curve:' + cv.name)) segs(cv.S, '#e9eef5', 1);
    if (layerVisible('axes')) {
      ctx.save(); ctx.setLineDash([6, 4]); ctx.strokeStyle = theme.axis; ctx.globalAlpha = 0.55;
      ctx.beginPath(); ctx.moveTo(px(0), py(m.view.z_min)); ctx.lineTo(px(0), py(m.view.z_max)); ctx.stroke(); ctx.restore();
    }
  }
  // pan / zoom for the fallback
  if (!webgl) {
    let drag = null;
    on(canvas, 'pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, cx: view2.cx, cy: view2.cy }; canvas.setPointerCapture?.(e.pointerId); userMoved.section = true; hideProbe(); });
    on(canvas, 'pointermove', (e) => {
      if (!drag) return;
      const ppu = ppu2 * view2.zoom;
      view2.cx = drag.cx - (e.clientX - drag.x) / ppu; view2.cy = drag.cy + (e.clientY - drag.y) / ppu; dirty = true;
    });
    on(canvas, 'pointerup', () => { drag = null; });
    on(canvas, 'wheel', (e) => {
      e.preventDefault();
      const rr = canvas.getBoundingClientRect(), mp = map2();
      const [wx, wy] = mp.toWorld(e.clientX - rr.left, e.clientY - rr.top);
      const f = Math.exp(-e.deltaY * 0.0015);
      view2.zoom = clamp(view2.zoom * f, 0.2, 400);
      const ppu = ppu2 * view2.zoom;
      view2.cx = wx - (e.clientX - rr.left - W / 2) / ppu; view2.cy = wy + (e.clientY - rr.top - H / 2) / ppu;
      userMoved.section = true; dirty = true;
    }, { passive: false });
  }

  /* -------------------------------------------------------------- probe */
  let probeReq = null;
  const raycaster = new THREE.Raycaster();
  const sampleAt = (r, z) => C.sampler(r, z);
  function pick3D(px, py) {
    const ndc = new THREE.Vector2((px / W) * 2 - 1, -(py / H) * 2 + 1);
    raycaster.setFromCamera(ndc, cam3);
    const o = raycaster.ray.origin, d = raycaster.ray.direction;
    let best = null;
    const dphi = (360 - state.cutaway) * DEG;
    if (layerVisible('cuts') && state.cutaway > 0) {
      for (const phi of [state.phi0, state.phi0 + dphi]) {
        const n = new THREE.Vector3(Math.sin(phi), -Math.cos(phi), 0), u = new THREE.Vector3(Math.cos(phi), Math.sin(phi), 0);
        const dn = d.dot(n);
        if (Math.abs(dn) < 1e-9) continue;
        const t = -o.dot(n) / dn;
        if (!(t > 0) || (best && t >= best.t)) continue;
        const p = o.clone().addScaledVector(d, t);
        const r = p.dot(u);
        if (r < 0) continue;
        const s = sampleAt(r, p.z);
        const w = C.clip3;
        const fieldShown = s.inField && layerVisible('field') && r <= w.r_max && p.z >= w.z_min && p.z <= w.z_max;
        const regionShown = s.region && layerVisible('region:' + s.region.name);
        if (fieldShown || regionShown) best = { t, sample: s, phi, onSurface: false };
      }
    }
    const meshes = C.surf.filter((m) => m.visible && m.geometry.attributes.position);
    if (meshes.length) {
      const hit = raycaster.intersectObjects(meshes, false)[0];
      if (hit && (!best || hit.distance < best.t)) {
        const p = hit.point, r = Math.hypot(p.x, p.y);
        const s = sampleAt(r, p.z);
        s.region = hit.object.userData.region;
        best = { t: hit.distance, sample: s, phi: Math.atan2(p.y, p.x), onSurface: true };
      }
    }
    return best;
  }
  function doProbe() {
    const req = probeReq;
    probeReq = null;
    if (!ui.probe || !req) return;
    let res = null;
    if (state.mode === '3d' && webgl) res = pick3D(req.x, req.y);
    else {
      const [x, y] = map2().toWorld(req.x, req.y);
      const s = sampleAt(Math.abs(x), y);
      if (s.inField || s.region) res = { sample: s, onSurface: false };
    }
    if (!res) { hideProbe(); return; }
    ui.probe.innerHTML = probeHTML(res);
    ui.probe.hidden = false;
    const pw = ui.probe.offsetWidth, ph = ui.probe.offsetHeight;
    let lx = req.x + 16, ly = req.y + 16;
    if (lx + pw > W - 6) lx = req.x - pw - 14;
    if (ly + ph > H - 6) ly = req.y - ph - 14;
    ui.probe.style.transform = `translate(${Math.round(Math.max(4, lx))}px,${Math.round(Math.max(4, ly))}px)`;
  }
  function hideProbe() { if (ui.probe) ui.probe.hidden = true; }
  const lenFmt = (x) => minus(x.toFixed(clamp(4 - Math.ceil(Math.log10(model.L || 1)), 1, 8)));
  function probeHTML(res) {
    const s = res.sample, f = model.field;
    let html = `<div class="c">r = ${lenFmt(s.r)} m &nbsp; z = ${lenFmt(s.z)} m`;
    if (res.phi != null && state.mode === '3d') html += ` &nbsp; φ = ${Math.round((((res.phi / DEG) % 360) + 360) % 360)}°`;
    html += '</div>';
    const R = s.region;
    if (R) {
      const st = roleStyle(R.role), col = R.role === 'superconductor' ? st.color : R.color;
      html += `<div class="r"><span class="axf-sw" style="background:${esc(col)}"></span>${esc(R.name)} <span class="o">${esc(st.label)}${R.material && R.material !== R.role ? ' · ' + esc(R.material) : ''}${res.onSurface ? ' · surface' : ''}</span></div>`;
    } else if (s.inField) {
      html += '<div class="r"><span class="axf-sw" style="background:transparent"></span>vacuum</div>';
    }
    if (R && R.role === 'superconductor' && !s.inField) {
      html += '<div class="n">B = 0 inside the superconductor (excluded from the field mesh).</div>';
      return html;
    }
    if (!f || !s.inField) {
      html += `<div class="n">${f ? 'Outside the sampled field window.' : 'No field data in this scene.'}</div>`;
      return html;
    }
    const q = state.quantity;
    if (q && Number.isFinite(s.values[q])) {
      html += `<div class="m">${richHTML(qmeta(q).sym)} = ${esc(fmtVal(s.values[q], f.units[q] || ''))}</div>`;
    }
    const extra = [];
    for (const k of ['Bmag', 'Br', 'Bz', 'Bphi', 'pickup_Bmag']) {
      if (k === q || !(k in s.values) || f.stats[k].zero) continue;
      extra.push(`${richHTML(qmeta(k).sym)} = ${esc(fmtVal(s.values[k], f.units[k] || ''))}`);
    }
    if (extra.length) html += `<div class="o">${extra.join(' &nbsp;·&nbsp; ')}</div>`;
    if (res.onSurface) html += '<div class="n">Field interpolated at the surface (outside side).</div>';
    return html;
  }
  // hover shows the probe; a click or tap (without dragging) pins it, a second
  // click on the pin (or Escape, or moving the view) releases it
  let pinned = null, downAt = null;
  function unpin() { if (pinned) { pinned = null; hideProbe(); dirty = true; } }
  function pinAt(x, y) { pinned = { x, y }; probeReq = { x, y }; dirty = true; }
  on(canvas, 'pointermove', (e) => {
    if (e.buttons || pinned) { if (e.buttons && !pinned) hideProbe(); return; }
    const rr = canvas.getBoundingClientRect();
    probeReq = { x: e.clientX - rr.left, y: e.clientY - rr.top };
  });
  on(canvas, 'pointerdown', (e) => { downAt = { x: e.clientX, y: e.clientY, t: performance.now() }; });
  on(canvas, 'pointerup', (e) => {
    if (!downAt || Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) > 5 || performance.now() - downAt.t > 600) { downAt = null; return; }
    downAt = null;
    const rr = canvas.getBoundingClientRect(), x = e.clientX - rr.left, y = e.clientY - rr.top;
    if (pinned && Math.hypot(x - pinned.x, y - pinned.y) < 10) unpin(); else pinAt(x, y);
  });
  on(canvas, 'pointerleave', () => { if (!pinned) { probeReq = null; hideProbe(); } });
  on(root, 'keydown', (e) => { if (e.key === 'Escape') unpin(); });

  /* ------------------------------------------------------------ actions */
  function chooseQuantity(q) {
    const f = model.field;
    if (!f || !f.names.length) return null;
    if (q && f.q[q]) return q;
    for (const k of ['Bmag', 'pickup_Bmag', ...f.names]) if (f.q[k] && !f.stats[k].zero) return k;
    return f.names[0];
  }
  function bgForScreenshot(ctx, w, hh) {
    const g = ctx.createRadialGradient(w / 2, hh * 0.32, 0, w / 2, hh * 0.32, Math.max(w, hh) * 0.75);
    g.addColorStop(0, theme.bgInner); g.addColorStop(1, theme.bg);
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, hh);
  }
  function screenshot() {
    renderNow();
    const w = canvas.width, hh = canvas.height;
    const out = doc.createElement('canvas');
    out.width = w; out.height = hh;
    const ctx = out.getContext('2d');
    bgForScreenshot(ctx, w, hh);
    ctx.drawImage(canvas, 0, 0);
    ctx.drawImage(overlay, 0, 0, w, hh);
    const rr = root.getBoundingClientRect();
    if (ui.cbarCanvas && ui.cbar.style.display !== 'none') {
      const cr = ui.cbar.getBoundingClientRect();
      ctx.save();
      ctx.globalAlpha = 0.92; ctx.fillStyle = theme.bg;
      ctx.fillRect((cr.left - rr.left) * dpr, (cr.top - rr.top) * dpr, cr.width * dpr, cr.height * dpr);
      ctx.restore();
      ctx.drawImage(ui.cbarCanvas, (cr.left - rr.left) * dpr, (cr.top - rr.top) * dpr, cr.width * dpr, cr.height * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = theme.fg; ctx.font = `650 16px ${FONT}`; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
    ctx.fillText(opts.title || model.name, 16, 14);
    if (state.quantity) {
      ctx.fillStyle = theme.fg2; ctx.font = `500 12px ${FONT}`;
      ctx.fillText(`${state.mode === '3d' ? '3D cutaway' : 'meridional section'} · axionFEM viewer`, 16, 35);
    }
    return out.toDataURL('image/png');
  }
  function savePNG() {
    try {
      const a = h('a', { href: screenshot(), download: `${(opts.title || model.name).replace(/[^\w.-]+/g, '_')}_${state.mode}.png` });
      doc.body.appendChild(a); a.click(); a.remove();
    } catch (e) { showMessage('Could not save the image: ' + e.message); }
  }
  let msgTimer = 0;
  function showMessage(text, ms = 0) {
    if (!ui.msg) return;
    ui.msg.textContent = text; ui.msg.hidden = !text;
    clearTimeout(msgTimer);
    if (ms) msgTimer = setTimeout(() => { ui.msg.hidden = true; }, ms);
  }
  function resolveLayer(name) {
    if (typeof name !== 'string') throw new Error('toggleLayer: layer name must be a string');
    if (/^(region|curve|lines):/.test(name)) return name;
    if (['field', 'surfaces', 'cuts', 'axes'].includes(name)) return name;
    if (model.regions.some((R) => R.name === name)) return 'region:' + name;
    if (model.curves.some((cv) => cv.name === name)) return 'curve:' + name;
    const alias = { static: 'lines:static', pickup: 'lines:pickup', fieldLines: 'lines:static', staticLines: 'lines:static', pickupLines: 'lines:pickup', field_lines: 'lines:static', cutFaces: 'cuts', planes: 'cuts', surface: 'surfaces' };
    if (alias[name]) return alias[name];
    throw new Error(`toggleLayer: unknown layer '${name}'. Known: ${layerNames().join(', ')}`);
  }
  function layerNames() {
    return ['field', 'lines:static', 'lines:pickup', 'surfaces', 'cuts', 'axes', ...model.regions.map((R) => 'region:' + R.name), ...model.curves.map((cv) => 'curve:' + cv.name)];
  }
  function setLayer(key, visible) {
    if (key.startsWith('lines:') && state.linesAuto) {
      for (const fam of ['static', 'pickup']) { if (lineFamilyVisible(fam)) state.hidden.delete('lines:' + fam); else state.hidden.add('lines:' + fam); }
      state.linesAuto = false;
    }
    const vis = visible ?? state.hidden.has(key);
    if (vis) state.hidden.delete(key); else state.hidden.add(key);
  }

  const api = {
    /** Colour the field map by a quantity of scene.field.quantities. */
    setQuantity(name) {
      const f = model.field;
      if (!f) return api;
      if (!f.q[name]) throw new Error(`setQuantity: unknown quantity '${name}'. Available: ${f.names.join(', ')}`);
      state.quantity = name;
      updateField(); applyVisibility(); syncControls();
      return api;
    },
    /** 'linear' or 'log' colour scale. */
    setScale(s) {
      if (s !== 'linear' && s !== 'log') throw new Error("setScale: use 'linear' or 'log'");
      state.scale = s; updateField(); syncControls(); return api;
    },
    /** Colormap name (see COLORMAPS) or 'auto'. */
    setColormap(name) {
      if (name !== 'auto' && !LUT_HEX[name]) throw new Error(`setColormap: unknown colormap '${name}'. Available: auto, ${COLORMAPS.join(', ')}`);
      state.colormap = name; updateField(); syncControls(); return api;
    },
    /** Fix the colour range [min, max] in the quantity's units; null for automatic. */
    setRange(lo, hi) {
      state.range = lo == null ? null : [Number(lo), Number(hi)];
      updateField(); return api;
    },
    /** '3d' (cutaway) or 'section' (2D meridional half-plane, mirrored). */
    setMode(mode) {
      const m = mode === 'section' || mode === '2d' ? 'section' : '3d';
      if (m === '3d' && !webgl) { showMessage('3D needs WebGL, which is unavailable here; showing the 2D section.', 4000); return api; }
      state.mode = m;
      if (webgl) { controls3.enabled = m === '3d'; controls2.enabled = m === 'section'; }
      unpin(); hideProbe(); applyVisibility(); syncControls();
      if (!userMoved[m]) fitView();
      dirty = true;
      return api;
    },
    /** Opening angle of the removed wedge in degrees (0 = full revolution; default 90). */
    setCutaway(deg) {
      state.cutaway = clamp(Number(deg) || 0, 0, 359);
      if (webgl) rebuildSurfaces();
      applyVisibility(); syncControls();
      return api;
    },
    /** Show/hide a layer: a region or curve name, 'field', 'lines:static', 'lines:pickup', 'surfaces', 'cuts', 'axes'. */
    toggleLayer(name, visible) {
      setLayer(resolveLayer(name), visible);
      applyVisibility(); syncControls(); drawColorbar();
      return api;
    },
    setTransparent(on2) { state.transparent = !!on2; applyTransparency(); syncControls(); return api; },
    setFieldBeyondDevice(on2) { state.beyond = !!on2; if (webgl) updateClipping(); syncControls(); dirty = true; return api; },
    setTheme(t) { state.theme = t === 'dark' || t === 'light' ? t : 'auto'; applyTheme(); return api; },
    /** Small status badge in the header (used by the live server); null clears it. */
    setStatus(text, kind = 'ok') {
      if (ui.badge) { ui.badge.hidden = !text; ui.badge.textContent = text || ''; ui.badge.dataset.kind = kind; }
      return api;
    },
    resetView() { fitView(); return api; },
    /** PNG data URL of the current view (with axes and colour bar). */
    screenshot,
    /** Replace the scene (e.g. after a re-solve), keeping view settings where possible. */
    setScene(newScene) {
      const next = prepareScene(newScene);
      const e0 = model.extent, e1 = next.extent;
      const same = Math.abs(e0.r_max - e1.r_max) < 0.05 * model.L && Math.abs(e0.z_min - e1.z_min) < 0.05 * model.L && Math.abs(e0.z_max - e1.z_max) < 0.05 * model.L;
      disposeContent(C);
      model = next;
      C = buildContent();
      state.quantity = chooseQuantity(state.quantity);
      if (webgl) { rebuildSurfaces(); applyTransparency(); }
      if (opts.ui) { fillHeader(); fillControls(); fillResults(); }
      updateField(); applyVisibility(); syncControls();
      if (!same) { userMoved['3d'] = false; userMoved.section = false; }
      if (!userMoved[state.mode]) fitView();
      renderNow();
      return api;
    },
    /** Sample the field at (r, z) in metres: {region, inField, values}. */
    probe(r, z) {
      const s = sampleAt(r, z);
      return { r, z, region: s.region ? s.region.name : null, inField: s.inField, values: s.values };
    },
    layers() { return layerNames().map((k) => ({ name: k, visible: layerVisible(k) })); },
    getState() {
      return { mode: state.mode, quantity: state.quantity, scale: state.scale, colormap: cmapName(), cutaway: state.cutaway, transparent: state.transparent, fieldBeyondDevice: state.beyond, range: mapping ? [mapping.lo, mapping.hi] : null, webgl, hidden: [...state.hidden] };
    },
    render() { renderNow(); return api; },
    /** Remove the viewer and free GPU resources. */
    dispose() {
      if (disposed) return;
      disposed = true;
      win.cancelAnimationFrame(raf);
      ro?.disconnect();
      for (const off of listeners) off();
      controls3?.dispose(); controls2?.dispose();
      disposeContent(C);
      lutTex.dispose(); envTex?.dispose(); axisMat.dispose(); axisMat2.dispose(); coneMat.dispose();
      if (webgl) { renderer.dispose(); renderer.forceContextLoss?.(); }
      root.remove();
    },
    get webgl() { return webgl; },
    get model() { return model; },
    VERSION,
  };

  /* --------------------------------------------------------- initialise */
  C = buildContent();
  state.quantity = chooseQuantity(opts.quantity);
  if (opts.quantity && model.field && !model.field.q[opts.quantity]) model.warnings.push(`quantity '${opts.quantity}' not in scene; showing '${state.quantity}'`);
  if (state.colormap !== 'auto' && !LUT_HEX[state.colormap]) state.colormap = 'auto';
  for (const n of opts.hide) { try { setLayer(resolveLayer(n), false); } catch (e) { model.warnings.push(e.message); } }
  for (const n of opts.show) { try { setLayer(resolveLayer(n), true); } catch (e) { model.warnings.push(e.message); } }
  buildUI();
  applyTheme();
  if (webgl) { rebuildSurfaces(); applyTransparency(); }
  updateField();
  applyVisibility();
  syncPanels();
  syncControls();
  if (root.clientWidth < 760) { state.panels.controls = false; state.panels.results = false; syncPanels(); }
  resize();
  fitView();
  if (!webgl) showMessage(`WebGL is unavailable (${glError ? glError.message : 'disabled'}); showing the 2D section (Canvas2D renderer).`, 0);
  renderNow();
  if (Array.isArray(opts.probeAt) && opts.probeAt.length === 2 && opts.probeAt.every(Number.isFinite)) {
    pinAt(opts.probeAt[0], opts.probeAt[1]);
    doProbe();
    renderNow();
  }
  const RO = win.ResizeObserver;
  const ro = RO ? new RO(() => { resize(); renderNow(); }) : null;
  if (ro) ro.observe(root); else on(win, 'resize', () => { resize(); renderNow(); });
  if (webgl) on(canvas, 'webglcontextlost', (e) => { e.preventDefault(); showMessage('The WebGL context was lost; reload the page.'); });
  raf = win.requestAnimationFrame(loop);
  return api;
}

export default createViewer;


/* ============================================================ 3D scenes */
function statsOf(a) {
  let mn = Infinity, mx = -Infinity, maxAbs = 0, minPos = Infinity;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (!Number.isFinite(v)) continue;
    if (v < mn) mn = v; if (v > mx) mx = v;
    const av = Math.abs(v);
    if (av > maxAbs) maxAbs = av;
    if (av > 0 && av < minPos) minPos = av;
  }
  if (!(mx >= mn)) { mn = 0; mx = 0; }
  return { min: mn, max: mx, maxAbs, minPos: Number.isFinite(minPos) ? minPos : 0, zero: maxAbs === 0, signed: mn < -1e-6 * maxAbs && mx > 1e-6 * maxAbs };
}

/** Decode a "symmetry": "3d" scene: region surfaces (x, y, z) and an optional slice. */
export function prepareScene3D(scene) {
  const warnings = [];
  const regions = (scene.regions || []).map((r, i) => {
    const w = `regions[${i}] ('${r.name}')`;
    const S = r.surface || {};
    const V = toF32(decodeArray(S.vertices, `${w}.surface.vertices`).data);
    const T = toU32(decodeArray(S.indices, `${w}.surface.indices`).data);
    checkIndices(T, V.length / 3, `${w}.surface.indices`);
    const B = S.Bmag ? toF32(decodeArray(S.Bmag, `${w}.surface.Bmag`).data) : null;
    const role = ROLE[r.role] ? r.role : 'vacuum';
    return { index: i, name: String(r.name ?? `region ${i}`), role, material: r.material || '', color: r.color || roleStyle(role).color, V, T, B };
  });
  // slices: the two half-planes of the quarter cutaway (older scenes: one "slice")
  const colorOf = new Map(regions.map((R) => [R.name, R.role === 'superconductor' ? roleStyle(R.role).color : R.color]));
  const attrNames = scene.attr_names || {};
  const rawSlices = Array.isArray(scene.slices) ? scene.slices : (scene.slice ? [scene.slice] : []);
  const slices = rawSlices.map((F, si) => {
    const w = `slices[${si}]`;
    const V = toF32(decodeArray(F.vertices, `${w}.vertices`).data);
    const T = toU32(decodeArray(F.indices, `${w}.indices`).data);
    checkIndices(T, V.length / 3, `${w}.indices`);
    const ra = F.region_attr ? toU32(decodeArray(F.region_attr, `${w}.region_attr`).data) : null;
    const q = {};
    for (const k of Object.keys(F.quantities || {})) {
      const a = toF32(decodeArray(F.quantities[k], `${w}.quantities.${k}`).data);
      if (a.length === V.length / 3) q[k] = a; else warnings.push(`${w} quantity '${k}' has the wrong length; skipped`);
    }
    // triangles without field values (superconductor interiors) are drawn in their material colour
    const first = q[Object.keys(q)[0]];
    const fieldT = [], matT = [], matC = [];
    for (let t = 0; t < T.length / 3; t++) {
      const a = T[3 * t], b = T[3 * t + 1], c = T[3 * t + 2];
      if (first && Number.isFinite(first[a]) && Number.isFinite(first[b]) && Number.isFinite(first[c])) fieldT.push(a, b, c);
      else { matT.push(a, b, c); matC.push(colorOf.get(attrNames[String(ra ? ra[t] : '')]) || ROLE.superconductor.color); }
    }
    return { V, q, fieldT: Uint32Array.from(fieldT), matT: Uint32Array.from(matT), matC, plane: F.plane || { normal: [0, 1, 0], offset: 0 }, units: F.units || {} };
  });
  let slice = null;
  if (slices.length) {
    const names = Object.keys(slices[0].q).filter((k) => slices.every((S) => S.q[k])).sort((a, b) => QORDER.indexOf(a) - QORDER.indexOf(b));
    const stats = {};
    for (const k of names) {
      const all = new Float32Array(slices.reduce((n, S) => n + S.q[k].length, 0));
      let o = 0;
      for (const S of slices) { all.set(S.q[k], o); o += S.q[k].length; }
      stats[k] = statsOf(all);
    }
    slice = { slices, names, stats, units: Object.assign({}, DEFAULT_UNITS, slices[0].units) };
  }
  const cutaway = scene.cutaway && Array.isArray(scene.cutaway.center) ? { x: +scene.cutaway.center[0], y: +scene.cutaway.center[1] } : null;
  const e = scene.extent3 || { min: [-1, -1, -1], max: [1, 1, 1] };
  return { dim3: true, name: String(scene.name || 'design'), regions, slice, cutaway, extent: e, summary: scene.summary || {}, warnings };
}

/**
 * Viewer for 3D scenes: region surfaces (optionally colored by the surface |B|), the device cut open
 * at the slice plane, and the slice colored by a field quantity. Returns an API object.
 */
export function createViewer3D(container, scene, options = {}) {
  const doc = container.ownerDocument || document;
  const win = doc.defaultView || window;
  injectCSS(doc);
  const model = prepareScene3D(scene);
  const prefersDark = !win.matchMedia || win.matchMedia('(prefers-color-scheme: dark)').matches;
  const state = {
    quantity: model.slice ? (model.slice.names.includes(options.quantity) ? options.quantity : model.slice.names[0]) : null,
    scale: options.scale === 'log' ? 'log' : 'linear', colormap: LUT_HEX[options.colormap] ? options.colormap : 'viridis',
    cut: options.cut !== false, slice: options.slice !== false, surfaceField: !!options.surfaceField, hidden: new Set(options.hide || []),
    theme: options.theme === 'light' || options.theme === 'dark' ? options.theme : (prefersDark ? 'dark' : 'light'),
  };
  const listeners = [];
  const on = (t, ev, fn) => { t.addEventListener(ev, fn); listeners.push(() => t.removeEventListener(ev, fn)); };
  const root = h('div', { class: 'axf-root', tabindex: '-1' });
  root.dataset.theme = state.theme;
  container.appendChild(root);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: !!options.preserveDrawingBuffer });
  renderer.setPixelRatio(Math.min(win.devicePixelRatio || 1, options.maxPixelRatio || 2));
  renderer.localClippingEnabled = true;
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.setClearColor(0x000000, 0);
  const canvas = renderer.domElement;
  canvas.classList.add('axf-canvas');
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', `axionFEM 3D view of ${model.name}`);
  root.appendChild(canvas);
  const overlay = h('canvas', { class: 'axf-overlay' });
  root.appendChild(overlay);
  const octx = overlay.getContext('2d');

  const scene3 = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(30, 1, 0.01, 1000);
  cam.up.set(0, 0, 1);
  scene3.add(cam);
  let envTex = null;
  try {
    const pm = new THREE.PMREMGenerator(renderer);
    const room = new RoomEnvironment();
    envTex = pm.fromScene(room, 0.04).texture;
    pm.dispose();
    scene3.environment = envTex;
    scene3.environmentIntensity = 0.85;
  } catch (e) { /* optional */ }
  scene3.add(new THREE.HemisphereLight(0xe8eef7, 0x3a3f47, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 1.4);
  key.position.set(-0.55, 0.75, 1.0);
  cam.add(key); cam.add(key.target); key.target.position.set(0, 0, -1);
  const controls = new OrbitControls(cam, canvas);
  controls.enableDamping = true; controls.dampingFactor = 0.12; controls.zoomToCursor = true;

  // geometry
  const e = model.extent, ctr = new THREE.Vector3((e.min[0] + e.max[0]) / 2, (e.min[1] + e.max[1]) / 2, (e.min[2] + e.max[2]) / 2);
  const L = Math.max(e.max[0] - e.min[0], e.max[1] - e.min[1], e.max[2] - e.min[2]) || 1;
  // quarter cutaway: {x > xc and y < yc} is removed (two planes, clipIntersection)
  const cx = model.cutaway ? model.cutaway.x : ctr.x, cy = model.cutaway ? model.cutaway.y : ctr.y;
  const clipPlanes = [new THREE.Plane(new THREE.Vector3(-1, 0, 0), cx), new THREE.Plane(new THREE.Vector3(0, 1, 0), -cy)];
  const lutTex = new THREE.DataTexture(lutTextureData(state.colormap), 256, 1, THREE.RGBAFormat);
  lutTex.colorSpace = THREE.SRGBColorSpace; lutTex.magFilter = lutTex.minFilter = THREE.LinearFilter; lutTex.needsUpdate = true;
  const disposables = [lutTex];
  let Bsurf = { min: Infinity, max: -Infinity };
  for (const R of model.regions) if (R.B && R.role !== 'vacuum' && R.role !== 'port') { const st = statsOf(R.B); Bsurf.min = Math.min(Bsurf.min, st.min); Bsurf.max = Math.max(Bsurf.max, st.max); }
  const meshes = model.regions.map((R) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(R.V, 3));
    g.setIndex(new THREE.BufferAttribute(R.T, 1));
    g.computeVertexNormals();
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(R.V.length), 3));
    const st = roleStyle(R.role);
    const mat = new THREE.MeshStandardMaterial({ color: R.role === 'superconductor' ? st.color : R.color, metalness: st.metal, roughness: st.rough, side: THREE.DoubleSide,
      transparent: (st.opacity ?? 1) < 1, opacity: st.opacity ?? 1, clippingPlanes: clipPlanes, clipIntersection: true });
    const m = new THREE.Mesh(g, mat);
    m.userData.region = R;
    scene3.add(m); disposables.push(g, mat);
    return m;
  });
  const sliceMeshes = [], matMeshes = [];
  if (model.slice) {
    for (const S of model.slice.slices) {
      const nv = S.V.length / 3;
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(S.V, 3));
      g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(2 * nv), 2));
      g.setIndex(new THREE.BufferAttribute(S.fieldT, 1));
      const mat = new THREE.MeshBasicMaterial({ map: lutTex, side: THREE.DoubleSide, toneMapped: false, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 });
      const mesh = new THREE.Mesh(g, mat);
      mesh.userData.slice = S;
      scene3.add(mesh); sliceMeshes.push(mesh); disposables.push(g, mat);
      if (S.matT.length) {  // superconductor cross-sections
        const pos = new Float32Array(S.matT.length * 3), col = new Float32Array(S.matT.length * 3);
        const c3 = new THREE.Color();
        for (let k = 0; k < S.matT.length; k++) {
          const id = S.matT[k];
          pos[3 * k] = S.V[3 * id]; pos[3 * k + 1] = S.V[3 * id + 1]; pos[3 * k + 2] = S.V[3 * id + 2];
          c3.set(S.matC[Math.floor(k / 3)]);
          col[3 * k] = c3.r; col[3 * k + 1] = c3.g; col[3 * k + 2] = c3.b;
        }
        const gm = new THREE.BufferGeometry();
        gm.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        gm.setAttribute('color', new THREE.BufferAttribute(col, 3));
        const mm = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 });
        const mmesh = new THREE.Mesh(gm, mm);
        scene3.add(mmesh); matMeshes.push(mmesh); disposables.push(gm, mm);
      }
    }
  }
  let mapping = null;
  function applyColors() {
    const bytes = lutBytes(state.colormap);
    lutTex.image.data.set(lutTextureData(state.colormap)); lutTex.needsUpdate = true;
    if (sliceMeshes.length && state.quantity) {
      mapping = makeMapping(model.slice.stats[state.quantity], state.quantity, state.scale, 4, null);
      for (const mesh of sliceMeshes) {
        const a = mesh.userData.slice.q[state.quantity], uv = mesh.geometry.getAttribute('uv');
        for (let i = 0; i < a.length; i++) { const t = mapping.map(a[i]); uv.array[2 * i] = Number.isFinite(t) ? t : 0; uv.array[2 * i + 1] = 0.5; }
        uv.needsUpdate = true;
      }
    }
    const sm = makeMapping({ min: Bsurf.min, max: Bsurf.max, maxAbs: Math.max(Math.abs(Bsurf.min), Math.abs(Bsurf.max)), minPos: Math.max(Bsurf.min, 0), signed: false }, 'Bmag', 'linear', 4, null);
    meshes.forEach((m) => {
      const R = m.userData.region, col = m.geometry.getAttribute('color');
      const useField = state.surfaceField && R.B && R.role !== 'vacuum' && R.role !== 'port';
      m.material.vertexColors = !!useField;
      m.material.color.set(useField ? '#ffffff' : (R.role === 'superconductor' ? roleStyle(R.role).color : R.color));
      if (useField) {
        for (let i = 0; i < R.B.length; i++) {
          const k = Math.round(255 * sm.map(R.B[i]));
          col.array[3 * i] = (bytes[3 * k] / 255) ** 2.2; col.array[3 * i + 1] = (bytes[3 * k + 1] / 255) ** 2.2; col.array[3 * i + 2] = (bytes[3 * k + 2] / 255) ** 2.2;
        }
        col.needsUpdate = true;
      }
      m.material.needsUpdate = true;
    });
  }
  function applyVisibility() {
    meshes.forEach((m) => {
      m.visible = !state.hidden.has(m.userData.region.name);
      m.material.clippingPlanes = state.cut ? clipPlanes : [];
    });
    for (const mesh of sliceMeshes) mesh.visible = state.slice && state.cut;
    for (const mesh of matMeshes) mesh.visible = state.cut;
  }

  // UI
  const sm = model.summary, c = sm.coupling || {};
  const bm = (sm.benchmarks || [])[0];
  const sub = [c.Psi_SI_H_T_m2 != null ? `Ψ ${fmtSI(c.Psi_SI_H_T_m2, 'H·T·m²', 3)}` : null, c.L_p_H != null ? `L_p ${fmtSI(c.L_p_H, 'H', 3)}` : null,
    c.W_mag_J != null ? `W ${fmtSI(c.W_mag_J, 'J', 3)}` : null, bm && bm.F_over_F_req != null ? `𝓕/𝓕_req ${fmtNum(bm.F_over_F_req, 3)} (${bm.name})` : null].filter(Boolean).join(' · ');
  const header = h('div', { class: 'axf-header' }, h('div', { class: 'axf-title axf-glass' }, h('div', { class: 'axf-name' }, options.title || model.name, h('small', null, '3D')), h('div', { class: 'axf-sub' }, sub)));
  root.appendChild(header);
  const sel = (items, cur, fn) => { const s = h('select', null, ...items.map(([v, t]) => { const o = h('option', { value: v }, t); if (v === cur) o.selected = true; return o; })); on(s, 'change', () => fn(s.value)); return s; };
  const chk = (label, cur, fn) => { const i = h('input', { type: 'checkbox' }); i.checked = cur; on(i, 'change', () => fn(i.checked)); return h('label', { style: { display: 'block', whiteSpace: 'nowrap' } }, i, ' ', label); };
  const panel = h('div', { class: 'axf-glass', style: { position: 'absolute', right: '12px', top: '12px', padding: '8px 10px', borderRadius: '12px', zIndex: 3, display: 'grid', gap: '4px' } });
  if (model.slice) {
    panel.appendChild(sel(model.slice.names.map((q) => [q, `${qmeta(q).text} — ${qmeta(q).name}`]), state.quantity, (v) => { state.quantity = v; applyColors(); dirty = true; }));
    panel.appendChild(sel([['linear', 'linear'], ['log', 'log']], state.scale, (v) => { state.scale = v; applyColors(); dirty = true; }));
  }
  panel.appendChild(sel(COLORMAPS.map((k) => [k, k]), state.colormap, (v) => { state.colormap = v; applyColors(); dirty = true; }));
  panel.appendChild(chk('cut open at the slice', state.cut, (v) => { state.cut = v; applyVisibility(); dirty = true; }));
  if (model.slice) panel.appendChild(chk('field slice', state.slice, (v) => { state.slice = v; applyVisibility(); dirty = true; }));
  panel.appendChild(chk('surface |B| on conductors', state.surfaceField, (v) => { state.surfaceField = v; applyColors(); dirty = true; }));
  root.appendChild(panel);
  const legend = h('div', { class: 'axf-glass', style: { position: 'absolute', left: '12px', bottom: '12px', padding: '8px 10px', borderRadius: '12px', zIndex: 3, maxHeight: '45%', overflow: 'auto' } });
  for (const R of model.regions) {
    const sw = h('span', { style: { display: 'inline-block', width: '10px', height: '10px', borderRadius: '2px', background: R.role === 'superconductor' ? roleStyle(R.role).color : R.color, marginRight: '6px' } });
    const i = h('input', { type: 'checkbox' }); i.checked = !state.hidden.has(R.name);
    on(i, 'change', () => { if (i.checked) state.hidden.delete(R.name); else state.hidden.add(R.name); applyVisibility(); dirty = true; });
    legend.appendChild(h('label', { style: { display: 'block', whiteSpace: 'nowrap' }, title: `${R.material} (${roleStyle(R.role).label})` }, i, ' ', sw, R.name));
  }
  root.appendChild(legend);

  function drawColorbar(Wpx, Hpx, dprr) {
    octx.setTransform(dprr, 0, 0, dprr, 0, 0);
    octx.clearRect(0, 0, Wpx, Hpx);
    const parts = [];
    if (state.surfaceField) parts.push({ title: '|B| on surfaces', lo: Bsurf.min, hi: Bsurf.max, unit: 'T', log: false });
    if (sliceMeshes.length && sliceMeshes[0].visible && mapping) parts.push({ title: `${qmeta(state.quantity).text} (slice)`, lo: mapping.lo, hi: mapping.hi, unit: model.slice.units[state.quantity] || '', log: mapping.log });
    const bytes = lutBytes(state.colormap), th = THEMES[state.theme];
    parts.forEach((pt, k) => {
      const w = 220, hh = 12, x0 = Wpx - w - 24, y0 = Hpx - 56 - k * 58;
      for (let i = 0; i < w; i++) { const j = Math.round(255 * i / (w - 1)); octx.fillStyle = `rgb(${bytes[3 * j]},${bytes[3 * j + 1]},${bytes[3 * j + 2]})`; octx.fillRect(x0 + i, y0, 1.5, hh); }
      octx.fillStyle = th.fg; octx.font = `600 12px ${FONT}`; octx.textAlign = 'left';
      octx.fillText(pt.title, x0, y0 - 6);
      octx.font = `11px ${FONT}`;
      octx.textAlign = 'left'; octx.fillText(fmtSI(pt.log ? pt.lo : pt.lo, pt.unit, 3), x0, y0 + hh + 13);
      octx.textAlign = 'right'; octx.fillText(fmtSI(pt.hi, pt.unit, 3) + (pt.log ? ' (log)' : ''), x0 + w, y0 + hh + 13);
    });
  }
  let dirty = true, Wc = 1, Hc = 1, raf = 0, disposed = false;
  function resize() {
    const r = root.getBoundingClientRect();
    Wc = Math.max(1, r.width); Hc = Math.max(1, r.height);
    renderer.setSize(Wc, Hc, false);
    const dprr = renderer.getPixelRatio();
    overlay.width = Math.round(Wc * dprr); overlay.height = Math.round(Hc * dprr);
    cam.aspect = Wc / Hc; cam.updateProjectionMatrix();
    dirty = true;
  }
  function resetView() {
    // look into the removed quadrant (x > xc, y < yc), as the PNG renderer (azimuth -60, elevation 25)
    const el = 25 * DEG, az = -60 * DEG;
    const dir = new THREE.Vector3(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el));
    const dist = 0.62 * L / Math.tan((cam.fov * DEG) / 2);
    cam.position.copy(ctr).addScaledVector(dir, dist);
    cam.near = dist / 100; cam.far = dist * 10; cam.updateProjectionMatrix();
    controls.target.copy(ctr); controls.update();
    dirty = true;
  }
  function frame() {
    if (disposed) return;
    raf = win.requestAnimationFrame(frame);
    if (controls.update()) dirty = true;
    if (!dirty) return;
    dirty = false;
    renderer.render(scene3, cam);
    drawColorbar(Wc, Hc, renderer.getPixelRatio());
  }
  on(controls, 'change', () => { dirty = true; });
  const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(resize) : null;
  if (ro) ro.observe(root); else on(win, 'resize', resize);
  resize(); applyColors(); applyVisibility(); resetView(); frame();
  function screenshot() {
    renderer.render(scene3, cam);
    const out = doc.createElement('canvas');
    out.width = canvas.width; out.height = canvas.height;
    const x = out.getContext('2d');
    x.fillStyle = THEMES[state.theme].bg; x.fillRect(0, 0, out.width, out.height);
    x.drawImage(canvas, 0, 0); x.drawImage(overlay, 0, 0);
    return out.toDataURL('image/png');
  }
  return {
    model, state,
    setQuantity(q) { if (model.slice && model.slice.q[q]) { state.quantity = q; applyColors(); dirty = true; } },
    setScale(s) { state.scale = s === 'log' ? 'log' : 'linear'; applyColors(); dirty = true; },
    setColormap(n) { if (LUT_HEX[n]) { state.colormap = n; applyColors(); dirty = true; } },
    setCutOpen(v) { state.cut = !!v; applyVisibility(); dirty = true; },
    setSurfaceField(v) { state.surfaceField = !!v; applyColors(); dirty = true; },
    toggleRegion(name, show) { if (show === false || (show === undefined && !state.hidden.has(name))) state.hidden.add(name); else state.hidden.delete(name); applyVisibility(); dirty = true; },
    resetView, screenshot,
    getState() { return { quantity: state.quantity, scale: state.scale, colormap: state.colormap, cut: state.cut, surfaceField: state.surfaceField, hidden: [...state.hidden] }; },
    dispose() {
      disposed = true; win.cancelAnimationFrame(raf); if (ro) ro.disconnect();
      listeners.forEach((f) => f()); controls.dispose();
      disposables.forEach((d) => d.dispose()); envTex?.dispose(); renderer.dispose(); root.remove();
    },
  };
}
