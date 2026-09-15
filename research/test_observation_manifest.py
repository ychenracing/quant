"""Exact observation preservation must not depend on filesystem iteration order."""
from pathlib import Path
import tempfile,json,hashlib,unittest
from research.admission_geometry import preserve_manifest


class ObservationManifestTests(unittest.TestCase):
    def test_original_bytes_are_recovered_independently_of_creation_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'observations';folder.mkdir()
            for name in ('z.csv','a.csv'):(folder/name).write_text(name)
            raw=(json.dumps({name:hashlib.sha256(name.encode()).hexdigest() for name in ('a.csv','z.csv')},indent=2)+'\n').encode()
            original=root/'original.json';original.write_bytes(raw);expected=hashlib.sha256(raw).hexdigest()
            preserve_manifest(folder,original,expected)
            self.assertEqual((folder/'MANIFEST.json').read_bytes(),raw)
            preserve_manifest(folder,original,expected)
            self.assertEqual((folder/'MANIFEST.json').read_bytes(),raw)

    def test_changed_missing_extra_payload_and_wrong_manifest_are_rejected(self):
        for failure in ('changed','missing','extra','wrong_manifest'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);folder=root/'observations';folder.mkdir();(folder/'a.csv').write_text('original')
                raw=json.dumps({'a.csv':hashlib.sha256(b'original').hexdigest()}).encode();original=root/'original.json';original.write_bytes(raw);digest=hashlib.sha256(raw).hexdigest()
                if failure=='changed':(folder/'a.csv').write_text('changed')
                elif failure=='missing':(folder/'a.csv').unlink()
                elif failure=='extra':(folder/'b.csv').write_text('extra')
                else:original.write_bytes(raw+b' ')
                with self.assertRaises(ValueError):preserve_manifest(folder,original,digest)
                self.assertFalse((folder/'MANIFEST.json').exists())

if __name__=='__main__':unittest.main()
